import enum
import logging
import getpass
import sys
import argparse
import os
import yaml


class PathTruncatingFormatter(logging.Formatter):

    width = 30

    def format(self, record):
        if 'pathname' in record.__dict__.keys():
            # truncate the pathname
            content = os.path.basename(record.pathname)
            content = f"{content}:{record.lineno}"
            if len(content) > PathTruncatingFormatter.width:
                content = '..{}'.format(content[-PathTruncatingFormatter.width+2:])
            else:
                content = "{:>{}}".format(content, PathTruncatingFormatter.width)
            record.pathname = content
        return super(PathTruncatingFormatter, self).format(record)


class ConfigError(Exception):
    def __init__(self, message):
        super().__init__(message)


class _Expansion:

    class Type(enum.Enum):
        CURLY = "{}"
        SQUARE = "[]"

    def __init__(self, raw):
        if raw[0] == _Expansion.Type.CURLY.value[0]:
            self._type = _Expansion.Type.CURLY
        elif raw[0] == _Expansion.Type.SQUARE.value[0]:
            self._type = _Expansion.Type.SQUARE
        else:
            raise ConfigError("expansion bracket type unsupported")
        self._name = raw[1:-1]

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return "EXP{}{}{}".format(self._type.value[0], self._name, self._type.value[1])


class _ParsedConfigValue:
    def __init__(self, parts):
        self._parts = parts

    def substitute(self):
        resolved = []
        for x in self._parts:
            if isinstance(x, _Expansion):
                env_value = os.environ.get(x.name, "")
                resolved.append(env_value)
            else:
                resolved.append(x)
        return "".join(resolved)

    def __repr__(self):
        return "ParsedConfigValue([{}])".format(self._parts)


# noinspection PyArgumentList
class _ParserState(enum.Enum):
    SCAN = enum.auto()
    ESCAPE_NEXT = enum.auto()
    BRACKET_OPEN = enum.auto()
    TOKEN = enum.auto()


def _parse_expansions(raw: str):

    if type(raw) in [int, float, bool]:
        return raw

    escape_char = '\\'
    rv = []
    state = _ParserState.SCAN
    current_token = None
    i = 0
    has_tokens = False
    while i < len(raw):
        c = raw[i]
        if state == _ParserState.SCAN:
            if c == escape_char:
                state = _ParserState.ESCAPE_NEXT
            elif c == '$':
                rv.append(c) # might get popped if bracket doesn't follow
                state = _ParserState.BRACKET_OPEN
            else:
                rv.append(c)
        elif state == _ParserState.ESCAPE_NEXT:
            rv.append(c)
            state = _ParserState.SCAN
        elif state == _ParserState.BRACKET_OPEN:
            if c == "{":
                rv.pop()   # pop previously consumed '$'
                state = _ParserState.TOKEN
                current_token = c
            else:
                i -= 1
                state = _ParserState.SCAN
        elif state == _ParserState.TOKEN:
            current_token += c
            if c == '}':
                state = _ParserState.SCAN
                rv.append(_Expansion(current_token))
                has_tokens = True
                current_token = None
        else:
            raise ConfigError("unhanded config interpolation state")
        i += 1

    if current_token is not None:
        raise ConfigError("incomplete expansion, for '{}'".format(current_token))

    # aggregate the result into either a string or _ParsedConfigValue
    return _ParsedConfigValue(rv) if has_tokens else "".join(rv)


def _recurse_apply(obj, fn):
    if isinstance(obj, dict):
        return {k: _recurse_apply(v, fn) for (k, v) in obj.items()}
    elif isinstance(obj, list):
        return [_recurse_apply(x, fn) for x in obj]
    else:
        return fn(obj)


def _resolve_by_env(value):
    if hasattr(value, "substitute"):
        return value.substitute()
    else:
        return value  # eg, for raw int, fixed string, etc


class SigEnv:

    class _NoDefault:
        pass

    NoDefault = _NoDefault()

    def __init__(self):
        self._is_prod = getpass.getuser() == "svc_quantpod"

        self._config = {}
        self._log_sigenv = self._is_prod

        logger = logging.getLogger()
        debug = False
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        if self._is_prod:
            formatter = PathTruncatingFormatter(
                "%(asctime)s.%(msecs)03d | %(pathname)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
            )
        else:
            formatter = PathTruncatingFormatter(
                "%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
            )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if not SigEnv.is_pycharm():
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", type=str, help="config file", required=False)
            args = parser.parse_args()
            if args.config:
                self._config = yaml.safe_load(open(args.config, "rb"))
                logging.info("reading config: {}".format(args.config))
                if self._config is None:
                    logging.warning("config file is empty")
                    self._config = {}

        self._raw_config = self._config
        # interpolate the entire config
        interpolated_config = SigEnv._interpolate_config(self._config)
        self._config = interpolated_config

    @staticmethod
    def _interpolate_config(raw):
        # parse the raw config, to identify expansions
        parsed = _recurse_apply(raw, _parse_expansions)
        # substitute the expansions
        resolved = _recurse_apply(parsed, _resolve_by_env)
        return resolved

    @staticmethod
    def is_pycharm():
        # attempt to detect if sigenv is running within a jupyter context
        return len(sys.argv) > 0 and (sys.argv[0].endswith("pydevconsole.py") or sys.argv[0].endswith("ipykernel_launcher.py"))

    def get(self, item, default=NoDefault):
        return SigEnv._get_item(self._config, item, item, default, self._log_sigenv)

    def __getitem__(self, item):
        return SigEnv._get_item(self._config, item, item, None, self._log_sigenv)

    @property
    def config(self):
        return self._config

    @property
    def raw_config(self):
        return self._raw_config

    @property
    def is_prod(self):
        return self._is_prod

    @staticmethod
    def dump_value(value):
        if value is None:
            return "None"
        elif isinstance(value, str):
            return "'{}'".format(value)
        else:
            return str(value)

    @staticmethod
    def _get_item(config, path, full_path, default, do_log: bool):
        tokens = path.split(".") if isinstance(path, str) else path
        token = tokens[0]
        if token not in config or config is None:
            if default != SigEnv.NoDefault:
                if do_log:
                    logging.info("sigenv['{}'] -> default:{}".format(full_path, SigEnv.dump_value(default)))
                return default
            else:
                raise ConfigError("config key '{}' not found, when resolving '{}'".format(token, full_path))
        if len(tokens) == 1:
            value = config[token]
            if do_log:
                logging.info("sigenv['{}'] -> config:{}".format(full_path, SigEnv.dump_value(value)))
            return value
        else:
            return SigEnv._get_item(config[token], tokens[1:], full_path, default, do_log)


sigenv = SigEnv()
