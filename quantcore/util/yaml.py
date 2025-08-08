import yaml
import yaml.reader
import yaml.scanner
import yaml.parser
import yaml.composer
import yaml.constructor
import yaml.resolver
import re


# Idea of this SafeResolver is to improve on the yaml.BaseResolver to not bring
# in the full feature set of the yaml.Resolver.  The problem with the
# yaml.Resolver is that is converts strings like "NO" and "YES" into bools,
# which can be a source of errors.  So below, we only setup a mapping from
# 'true' and 'false' to be boolean.
class SafeResolver(yaml.resolver.BaseResolver):
    pass


SafeResolver.add_implicit_resolver(
        'tag:yaml.org,2002:bool',
        re.compile(r'''^(?:true|false)$''', re.X),
        list('tf'))

# Copied from yaml/resolver.py
SafeResolver.add_implicit_resolver(
        'tag:yaml.org,2002:float',
        re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
                    |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$''', re.X),
        list('-+0123456789.'))

# Copied from yaml/resolver.py
SafeResolver.add_implicit_resolver(
        'tag:yaml.org,2002:int',
        re.compile(r'''^(?:[-+]?0b[0-1_]+
                    |[-+]?0[0-7_]+
                    |[-+]?(?:0|[1-9][0-9_]*)
                    |[-+]?0x[0-9a-fA-F_]+
                    |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$''', re.X),
        list('-+0123456789'))


# Idea of this SafeLoader is to use the SaveResolver (defined above) to avoid
# pitfalls regarding bools, but also to detect duplicates in the yaml file.
class SafeLoader(yaml.reader.Reader,
                 yaml.scanner.Scanner,
                 yaml.parser.Parser,
                 yaml.composer.Composer,
                 yaml.constructor.SafeConstructor,
                 SafeResolver):

    def __init__(self, stream):
        yaml.reader.Reader.__init__(self, stream)
        yaml.scanner.Scanner.__init__(self)
        yaml.parser.Parser.__init__(self)
        yaml.composer.Composer.__init__(self)
        yaml.constructor.SafeConstructor.__init__(self)
        SafeResolver.__init__(self)

    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"YAML contains duplicate key: {key!r}")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def yaml_safe_load(stream):
    return yaml.load(stream, SafeLoader)
