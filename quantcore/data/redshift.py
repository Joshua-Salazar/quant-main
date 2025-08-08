import getpass

from ..util.auth import get_credentials

def get_redshift_creds(profile=None):
    if profile is None:
        profile = getpass.getuser()
    creds = get_credentials(f'redshift_{profile}', url_pass=False)
    return creds
