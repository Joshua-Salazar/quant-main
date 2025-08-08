# -*- coding: utf-8 -*-
import errno
import hashlib
import logging
import os
import tempfile
from time import time

from flask_caching.backends.filesystemcache import FileSystemCache

try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle  # type: ignore


logger = logging.getLogger(__name__)


class FileSystemCacheNotOwner(FileSystemCache):

    """Only difference with original FileSystemCache is it does not try to change permissions
    """

    def set(self, key, value, timeout=None, mgmt_element=False):
        result = False

        # Management elements have no timeout
        if mgmt_element:
            timeout = 0

        # Don't prune on management element update, to avoid loop
        else:
            self._prune()

        timeout = self._normalize_timeout(timeout)
        filename = self._get_filename(key)
        try:
            fd, tmp = tempfile.mkstemp(
                suffix=self._fs_transaction_suffix, dir=self._path
            )
            with os.fdopen(fd, "wb") as f:
                pickle.dump(timeout, f, 1)
                pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
            is_new_file = not os.path.exists(filename)
            os.replace(tmp, filename)
            # os.chmod(filename, self._mode) # remove this line
        except (IOError, OSError) as exc:
            logger.error("set key %r -> %s", key, exc)
        else:
            result = True
            logger.debug("set key %r", key)
            # Management elements should not count towards threshold
            if not mgmt_element and is_new_file:
                self._update_count(delta=1)
        return result
