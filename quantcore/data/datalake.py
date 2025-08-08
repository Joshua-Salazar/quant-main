from datetime import datetime, timedelta
import json

from warnings import warn
from urllib.error import HTTPError, URLError
import urllib.request

import pandas as pd
import requests
import os
import base64
import re


class Datalake:
    """
    Sample code that interacts with the datalake using Python 3
    @author Jimmy
    """

    def __init__(self):
        warn("This copy of Datalake will be deprecated in the near future. "
             "Please reference it in a version of the official Datalake git repo: "
             "https://gitea.capstoneco.com/DataEngineering/datalake.git",
             DeprecationWarning)
        self.url = ""
        self.url_base = "https://api.capstoneco.com/datalake/marketdata-api?"

    def getUploadURL(self, source, user, token, path, bucket=None, target_folder=None):
        ## get actual filename from path
        filename = (path.split('/')[-1]).split('.')[0]
        filename = re.sub('[^a-zA-Z0-9 \n\.]', '_', filename)

        if bucket and target_folder:
            self.url = self.url_base + "source="+source+"&user="+user+"&upload-token=yes&token="+token+"&bucket="+bucket+"&target-folder="+target_folder+'&filename='+filename
        else:
            self.url = self.url_base + "source="+source+"&user="+user+"&upload-token=yes&token="+token+'&filename='+filename

        try:
            response = urllib.request.urlopen(self.url)
        except HTTPError as e:
            print('Error code: ', e.code)
        except URLError as e:
            print('Reason: ', e.reason)
        else:
            data = json.loads(response.read())
            return data

    def putData(self, source, path, user, token, bucket=None, target_folder=None):
        if  os.path.isfile(path):
            url=self.getUploadURL(source, user, token, path, bucket, target_folder)

            if url:
                d=open(path).read()
                upload_status=requests.put(url, data=open(path).read())

                if upload_status.status_code==200:
                    print ("Upload successful! ")
                    return 0
                else:
                    print ("Upload FAIL")
                    print ("status: ", upload_status.status_code)
                    return -1

        else:
            print ("-----------------")
            print ("File:", path, " does not exist. Please check again.")
            return -1

    def getTickers(self, inTicker, outTicker, start=None, end=None, extra_fields=None, extra_values=None, format='str'):
        source="CTP_SECURITY_MASTER"
        if start is None:
            start=datetime(1970,1,1)

        if end is None:
            end=datetime(2099,1,1)

        self.url = self.url_base + '&source=' + source + '&ticker=' + inTicker + '&start='+start.strftime("%Y-%m-%d_%H:%M:%S")+'&end='+end.strftime("%Y-%m-%d_%H:%M:%S")+'&fields='+outTicker

        if extra_fields is not None and extra_values is not None:
            extra_fields = extra_fields.split("|")
            extra_values = extra_values.split("|")
            self.url += '&extra-fields=' + ",".join(extra_fields)
            zip_object = zip (extra_fields, extra_values)
            for (extra_field, extra_value) in zip_object:
                self.url += '&X' + extra_field + '=' + extra_value

        self.url = self.url.replace(" ", "%20")
        #print self.url
        try:
            #response = urllib.urlopen(self.url)
            response = urllib.request.urlopen(self.url)

        except HTTPError as e:
            print('Error code: ', e.code)
        except URLError as e:
            print('Reason: ', e.reason)
        else:
            data = json.loads(response.read())
            if data.get('body'):
                df = pd.read_json(data.get('body'), orient='records')  # type: pd.DataFrame
                columns = json.loads(data['column_names'])

                if len(df.columns) == len(columns):
                    df.columns = columns
                elif len(df.columns) < len(columns):
                    df.columns = columns[:-1]
                elif len(df.columns) > len(columns):
                    if df[df.columns[-1]].dtype == object:
                        df.drop(df.columns[-1], axis=1, inplace=True)
                        df.columns = columns

                if format == 'str':
                    df[outTicker] = df[outTicker].astype(str)

                    return ','.join(df[outTicker])
                else:
                    return df


    def getAutocomplete(self, source, index, previous_value=None):
        pv_64 = ''
        if previous_value:
            pv = previous_value.split("|")
            if len(pv) > 1:
                pv_64 =  ','.join( map(lambda x: base64.b64encode((x.encode())).decode('utf-8'), pv))
            else:
                pv_64 = base64.b64encode(previous_value.encode()).decode('utf-8')
        self.url = self.url_base + '&source=' + source + '&index-autocomplete=' + index + '&autocomplete-previous=' + pv_64

        #print ("url: ", self.url)

        try:
            #response = urllib.urlopen(self.url)
            response = urllib.request.urlopen(self.url)
        except HTTPError as e:
            print('Error code: ', e.code)
        except URLError as e:
            print('Reason: ', e.reason)
        else:
            data = json.loads(response.read())
            return data



    def getAvailableFields(self, source):
        """
        Get available fields.
        :param source:
        :type source: str
        :rtype list
        """
        self.url = self.url_base + '&source=' + source + '&available_fields=yes'
        self.url = self.url.replace(" ", "%20")

        # print self.url
        try:
            response = urllib.request.urlopen(self.url)
        except HTTPError as e:
            print('Error code: ', e.code)
        except URLError as e:
            print('Reason: ', e.reason)
        else:
            data = json.loads(response.read())
            return eval(data.get('fields'))

    def getData(self, source, ticker, fields=None, start=None, end=None, period=None, extra_fields=None,
                extra_values=None):
        """
        :type source: str
        :type ticker: str
        :type fields: str or list
        :type start: datetime
        :type end: datetime
        :type period: datetime
        """
        if start is not None and end is None and isinstance(period, int):
            end = start + timedelta(days=period)

        elif start is None and end is not None and isinstance(period, int):
            start = end - timedelta(days=period)

        #print ("start: ", start, " end: ", end)

        if not start and not end:
            return None

        if fields is None:
            fields = self.getAvailableFields(source)
            fields = ','.join(fields)
        elif isinstance(fields, list):
            fields = ','.join(fields)
        elif isinstance(fields, str):
            if fields.lower() == 'all' or fields.lower() == 'none':
                fields = self.getAvailableFields(source)
                fields = ','.join(fields)

        endStr = end.strftime("%Y-%m-%d_%H:%M:%S")
        startStr = start.strftime("%Y-%m-%d_%H:%M:%S")

        _url = self.url_base + '&source=' + source + '&ticker=' + ticker + '&start=' + startStr + '&end=' + endStr + '&fields=' + fields
        self.url = _url.replace(" ", "%20")

#        if extra_fields is not None and extra_values is not None:
#            self.url += '&extra-fields=' + extra_fields + '&X' + extra_fields + '=' + extra_values

        if extra_fields is not None and extra_values is not None:
            extra_fields = extra_fields.split("|")
            extra_values = extra_values.split("|")
            self.url += '&extra-fields=' + ",".join(extra_fields)
            zip_object = zip (extra_fields, extra_values)
            for (extra_field, extra_value) in zip_object:
                self.url += '&X' + extra_field + '=' + extra_value

        self.url = self.url.replace(" ", "%20")

        #print (self.url)

        if ("TICK" in source and "TICKDATA" not in source and ((end-start).total_seconds()) > 0.5*60*60):
            mid = start + (end - start) / 2
            mid_plus_1_sec = mid + timedelta(seconds=1)
            #print "in tick, end minus start days", (end-start).seconds, "new mid: ", mid
            return (self.getData(source, ticker, fields, start, mid)).append(self.getData(source, ticker, fields, mid_plus_1_sec, end))
        elif ("MINUTE" in source and ((end-start).days) > 90):
            mid = start + (end - start) / 2
            mid_plus_1_sec = mid + timedelta(seconds=1)
            return (self.getData(source, ticker, fields, start, mid)).append(self.getData(source, ticker, fields, mid_plus_1_sec, end))
        else:
            try:
                response = urllib.request.urlopen(self.url)
            except Exception as e:
                if e.code == 504 or e.code == 502:
                    halfdays = int((end - start).days / 2)
                    mid = start + timedelta(days=halfdays)
                    mid_plus_1_sec = mid + timedelta(seconds=1)

                    ## add handling for recursive calls that have extra fields and values
                    ## print "request size too large, split into 2 calls"
                    if extra_fields:
                        ret = self.getData(source, ticker, fields, start, mid , extra_fields='|'.join(map(str, extra_fields)) ,extra_values='|'.join(map(str, extra_values)))
                    else:
                        ret = self.getData(source, ticker, fields, start, mid )

                    if ret is None:
                        if extra_fields:
                            return self.getData(source, ticker, fields, mid_plus_1_sec, end, extra_fields='|'.join(map(str, extra_fields)) ,extra_values='|'.join(map(str, extra_values)))
                        else:
                            return self.getData(source, ticker, fields, mid_plus_1_sec, end)
                    else:
                        if extra_fields:
                            return ret.append(self.getData(source, ticker, fields, mid_plus_1_sec, end, extra_fields='|'.join(map(str, extra_fields)) ,extra_values='|'.join(map(str, extra_values))), ignore_index=True)
                        else:
                            return ret.append(self.getData(source, ticker, fields, mid_plus_1_sec, end))

                else:
                    print("unknown api error: ", e)

            else:
                data = json.loads(response.read())
                if data.get('body'):
                    df = pd.read_json(data.get('body'), orient='records')  # type: pd.DataFrame
                    columns = json.loads(data['column_names'])

                    if len(df.columns) == len(columns):
                        df.columns = columns
                    elif len(df.columns) < len(columns):
                        df.columns = columns[:-1]
                    elif len(df.columns) > len(columns):
                        if df[df.columns[-1]].dtype == object:
                            df.drop(df.columns[-1], axis=1, inplace=True)
                            df.columns = columns

                    size = (data['size'])
                    return df
                elif data.get('errorType'):
                    print('ERROR: authentication failed. Please check authentication token.')
                elif data.get('errorMessage') or data.get('message'):
                    #print ("error message: ", data)
                    #if data['message'] == "Internal server error" or data['errorMessage'] == "body size is too long":
                    if data.get('message') == "Internal server error" or data.get('errorMessage') == "body size is too long":
                        halfdays = int((end - start).days / 2)
                        mid = start + timedelta(days=halfdays)
                        mid_plus_1_sec = mid + timedelta(seconds=1)
                        return self.getData(source, ticker, fields, start, mid).append(
                            self.getData(source, ticker, fields, mid_plus_1_sec, end), ignore_index=True)
                    else:
                        print("unknown api error: ", data['errorMessage'])
                        return data['errorMessage']
                elif data.get("rows") == 0:
                    return pd.DataFrame()
                else:
                    print("unknown api error: ", data)
                    return data
