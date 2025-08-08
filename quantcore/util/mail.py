from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib
import getpass
import os
import logging


def split_mailto(s: str):
    return [x.strip() for x in s.split(",")]


def send_email(send_to, subject, body, attach_files=None, body_mime_type="html",
               cc_to=None, shorten_filename=False):
    """Send an email to a list of recipients, with optional attachements. If sending
     a plain file, provide body_mime_type='plain'. The `send_to` and `cc_to` can
     be a list or comma-separated string. Set `shorten_filename` to remove the
     path from filenames as they appears in the email display.

    """

    if isinstance(send_to, str):
        send_to = split_mailto(send_to)
    if isinstance(cc_to, str):
        cc_to = split_mailto(cc_to)
    assert isinstance(send_to, list)

    msg = MIMEMultipart("alternative")
    my_email = getpass.getuser() + "@capstoneco.com"
    msg["From"] = my_email
    msg["To"] = ", ".join(send_to)
    msg["Subject"] = subject
    if cc_to is not None:
        msg["Cc"] = ", ".join(cc_to)

    # add body
    if body is not None:
        body += "<p>For internal use only; not for external distribution."
        msg.attach(MIMEText(body, body_mime_type))
    elif attach_files is not None:
        msg.attach(MIMEText("See attached."))

    # attach files
    if attach_files is not None:
        if not isinstance(attach_files, list):
            attach_files = [attach_files]
        for filename in attach_files:
            with open(filename, "rb") as f:
                display_filename = filename
                if shorten_filename:
                    display_filename = os.path.basename(filename)
                part = MIMEApplication(f.read(), Name=display_filename)
                part["Content-Disposition"] = "attachment; filename={}".format(display_filename)
                msg.attach(part)

    logging.info("sending email, subject: '{}', to: '{}'".format(subject, send_to))
    smtp_conn = smtplib.SMTP("usmrtr.capstoneco.com", 25)
    smtp_conn.sendmail(my_email, send_to, msg.as_string())
    smtp_conn.close()
