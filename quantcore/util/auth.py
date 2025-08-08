import pathlib
import os
import json
import platform
import urllib
from typing import Dict


def get_credentials(user) -> Dict:
    """
    Get user credentails from the default quant secrets folder.
    @param user: identifies the base filename (without json suffix) of the credentials file.
    @param url_pass: if the username & password of the credentials are to be used for a web
                     request, they will be quoted using urlib.
    @return: dict of credentials, with optional url quoting
    """
    fn = pathlib.Path.home().joinpath(".quant_credentials/{}.json".format(user))
    try:
        if platform.system() == "Linux":
            stat = os.stat(fn).st_mode & 0o077
            if stat != 0:
                raise Exception("please chmod profile-file '{}' to be 700".format(fn))
        with open(fn) as f:
            creds = json.load(f)
            return creds
    except FileNotFoundError:
        raise Exception("credentials file for user '{}' not found, expected at '{}'".format(user, fn))
