# import Python packages required for the notebook
# used to generate URL-encoded parameters
import datetime
from urllib.parse import (
    parse_qs,
    quote,
    urlencode,
)

import pandas as pd
import requests  # run `pip install requests` if error shows package not installed

dataquery_config = {
    "client_id": "uRFZhSH8AwEvpXM8",  # TODO: change required
    "client_secret": "Fji5mq2qwkhglc4s9ngwwZppfapub1hioHtbz2fvR45MakEqsxbasyzrarAfn77834de5t8qSabvsgX",  # TODO: change required
    # 'https_proxy': 'REPLACE_WITH_YOUR_PROXY_AND_UNCOMMENT_IF_REQUIRED' #TODO: only change when you need a proxy to connect to DataQuery endpoints
}

# Utility functions and variables for the notebook

_dq_stored_token = None  # singleton-like variable


def _get_token(dataquery_config: dict) -> str:
    """
    Generating an access token for DataQuery APIs access via OAuth using Client ID and Secret. The function also handles token refresh when the access token is nearly expiring.

    Parameters
    ----------
    `dataquery_config` : dict
        The dictionary with the properties needed for the connection

    Raises
    ----------
    Exception:
        When unable to obtain an access token

    Returns
    -------
    `str`
        The access token for API connection
    """
    global _dq_stored_token  # reference to module variable
    dq_token_provider_url = (
        "https://authe.jpmchase.com/as/token.oauth2"  # JP Morgan OAuth server
    )
    dq_api_resource_id = (
        "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"  # DataQuery APIs resource ID
    )

    # only generate an access token when
    # 1) there's no existing access token stored in `_dq_stored_token`
    # 2) the existing access token reaches 85% of its expiration duration
    if _dq_stored_token is None or (
        ((datetime.datetime.now() - _dq_stored_token["created_at"]).total_seconds())
        >= (_dq_stored_token["expires_in"] * 0.85)
    ):
        json = requests.post(
            url=dq_token_provider_url,
            proxies={"https": dataquery_config["https_proxy"]}
            if "https_proxy" in dataquery_config
            else {},
            data={
                "grant_type": "client_credentials",
                "client_id": dataquery_config["client_id"],
                "client_secret": dataquery_config["client_secret"],
                "aud": dq_api_resource_id,
            },
        ).json()
        # stores the token in the singleton-like variable together with `created_at` (=system date-time) and `expires_in` (=expiration time in seconds, from the response)
        if "access_token" in json:
            _dq_stored_token = {
                "created_at": datetime.datetime.now(),
                "access_token": json["access_token"],
                "expires_in": json["expires_in"],
            }
        else:
            raise Exception("Unable to obtain an OAuth token: {0}".format(json))

    return _dq_stored_token["access_token"]


def get_dq_api_result(path: str, params: dict, page_count: int = 1) -> dict:
    """Getting data from DataQuery APIs

    Args:
        path (str): The DataQuery APIs endpoint for the request

        params (dict): The parameters for the API request. See [API specification](https://developer.jpmorgan.com/products/dataquery_api/specification) for all supported parameters.

        page_count (int, optional): The count of pages to request when results are paginated. Defaults to 1. `0` = all pages.

    Returns:
        dict: The API response converted to dictionary
    """

    if page_count < 0:
        raise Exception("page_count must be equal to or greater than 0")

    dq_api_url = "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"  # DataQuery APIs base url

    # concatenates path to base path from config, and issues a GET call using the params and including the token from function above
    token = _get_token(dataquery_config)
    # returns the JSON response as dict if successful or raises an exception if error

    response = requests.get(
        url=dq_api_url + path,
        params=urlencode(params, True, quote_via=quote),
        headers={"Authorization": "Bearer " + token},
        proxies={"https": dataquery_config["https_proxy"]}
        if "https_proxy" in dataquery_config
        else {},
    )
    # print(response.request.url) # uncomment this line if you want to see the constructed URL for API call
    response_dict = response.json()

    if "errors" in response_dict:
        raise Exception("Unable to obtain response: {0}".format(response_dict))

    if "info" in response_dict:
        raise Exception("{0}".format(response_dict))

    if "/grid-data" in path:
        return response_dict

    if page_count == 1:
        if (
            response_dict.get("page-size") < response_dict.get("items")
            and response_dict["links"][1]["next"] is not None
        ):
            print(
                "{CBEG}Info: Current request retrieves {current} items from 1 page. DataQuery has {total} items related to this request. Add `page_count=0` to `get_dq_api_result()` request to retrieve all pages.{CEND}".format(
                    CBEG="\033[93m",
                    current=response_dict.get("page-size"),
                    total=response_dict["items"],
                    CEND="\033[00m",
                )
            )
            try:
                next_page_token = parse_qs(response_dict["links"][1]["next"]).get(
                    "page"
                )[0]
                print(
                    "{CBEG}Token for next page that can be passed as `page` in `params` for the `get_dq_api_result()` request: {BBEG}{token}{CEND}".format(
                        CBEG="\033[93m",
                        BBEG="\033[44m",
                        token=next_page_token,
                        CEND="\033[00m",
                    )
                )
            except Exception:
                print(
                    "{CBEG}This is the last page.{CEND}".format(
                        CBEG="\033[93m", CEND="\033[00m"
                    )
                )
        return response_dict

    if page_count == 0 or page_count >= 2:
        try:
            next_path = response_dict["links"][1]["next"]
            paginated_response = []
            paginated_response.append(response_dict)
            get_next = True
            retrieved_page_count = 1
            retrieved_item_count = response_dict.get("page-size")
            print(
                f'{min(page_count, (-(-response_dict["items"] // 20)))} pages requested{" instead" if (-(-response_dict["items"] // 20)) < page_count else ""}. DataQuery total: {(-(-response_dict["items"] // 20))} pages for {response_dict["items"]} items.\nReceived total: {retrieved_page_count} pages for {retrieved_item_count} items',
                end="\r",
                flush=True,
            )
            while next_path is not None and get_next is True:
                next_page_response = requests.get(
                    url=dq_api_url + next_path,
                    headers={"Authorization": "Bearer " + token},
                    proxies={"https": dataquery_config["https_proxy"]}
                    if "https_proxy" in dataquery_config
                    else {},
                )
                next_page_response_dict = next_page_response.json()
                if "errors" in next_page_response_dict:
                    raise Exception(
                        "Unable to obtain response: {0}".format(next_page_response_dict)
                    )
                paginated_response.append(next_page_response_dict)
                retrieved_item_count += next_page_response_dict.get("page-size")
                retrieved_page_count += 1
                print(
                    "Page {} retrieved: {} items retrieved".format(
                        retrieved_page_count, retrieved_item_count
                    ),
                    end="\r",
                )
                get_next = (
                    False
                    if retrieved_page_count == page_count and page_count != 0
                    else True
                )
                next_path = next_page_response_dict["links"][1]["next"]
        except Exception:
            pass

        if retrieved_item_count < response_dict.get("items"):
            print(
                "{CBEG}Info: Current request retrieves {current} items from {pages} pages. DataQuery has {total} items related to this request. Add `page_count=0` to `get_dq_api_result()` request to retrieve all pages.{CEND}".format(
                    CBEG="\033[93m",
                    current=retrieved_item_count,
                    pages=retrieved_page_count,
                    total=response_dict["items"],
                    CEND="\033[00m",
                )
            )
            try:
                next_page_token = parse_qs(next_path).get("page")[0]
                print(
                    "{CBEG}Token for next page that can be passed as `page` in `params` for the `get_dq_api_result()` request: {BBEG}{token}{CEND}".format(
                        CBEG="\033[93m",
                        BBEG="\033[44m",
                        token=next_page_token,
                        CEND="\033[00m",
                    )
                )
            except Exception:
                print(
                    "{CBEG}This is the last page.{CEND}".format(
                        CBEG="\033[93m", CEND="\033[00m"
                    )
                )
        else:
            print("\nAll {} items received".format(retrieved_item_count))

        return paginated_response


def get_time_series_by_expressions(start_date, end_date, expressions):
    expression_results = get_dq_api_result(
        path="/expressions/time-series",
        params={
            "expressions": expressions,  # up to 20 expressions are supported - see API spec
            "start-date": start_date.strftime("%Y%m%d"),
            # Feb 20 2022 in `YYYYMMDD`format. Max one-year data per API request is recommended for best response time.
            "end-date": end_date.strftime("%Y%m%d"),
            # Mar 20 2022 in `YYYYMMDD` format. Max one-year data per API request is recommended for best response time.
            "data": "ALL",
            "calendar": "CAL_WEEKDAYS",
            "non-treatment": "NA_NOTHING",
        },
    )
    # normalize the JSON response at expression level
    df = pd.json_normalize(expression_results, record_path=["instruments", "attributes"])
    return df


def get_time_series_by_instruments(start_date, end_date, instruments, attributes):
    expression_results = get_dq_api_result(
        path="/instruments/time-series",
        params={
            "attributes": attributes,
            "instruments": instruments,
            "start-date": start_date.strftime("%Y%m%d"),
            # Feb 20 2022 in `YYYYMMDD`format. Max one-year data per API request is recommended for best response time.
            "end-date": end_date.strftime("%Y%m%d"),
            # Mar 20 2022 in `YYYYMMDD` format. Max one-year data per API request is recommended for best response time.
            "data": "ALL",
            "calendar": "CAL_WEEKDAYS",
            "non-treatment": "NA_NOTHING",
        },
    )
    # normalize the JSON response at expression level
    df = pd.json_normalize(expression_results, record_path=["instruments", "attributes"])
    return df


if __name__ == '__main__':
    df = get_time_series_by_expressions(datetime.datetime(2024, 8, 29), datetime.datetime(2024, 9, 4), [
        "FCRV_SOFR_SWAP_ZERO_1D_RT_MID",
    ])
    print(df)
