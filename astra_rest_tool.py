import requests
from urllib.parse import urlencode
import os
import json
ASTRA_URL = f'{os.environ["ASTRA_API_ENDPOINT"]}/api/rest/v2/keyspaces/{os.environ["ASTRA_KEYSPACE"]}'

# For the connection with CQL Tables, we will leverage the AstraDB REST API.
def astra_rest(table, pk, params={}, filters=[], method='GET', data={}):
    headers = {'Accept': 'application/json',
               'X-Cassandra-Token': f'{os.environ["ASTRA_TOKEN"]}'}
    url = f'{ASTRA_URL}/{table}/{"/".join(pk)}?{urlencode(params)}'

    res = requests.request(
        method, url=url, headers=headers, data=json.dumps(data))

    if int(res.status_code) >= 400:
        return res.text

    try:
        res_data = res.json()
        return res_data
    except ValueError:
        res_data = res.status_code
        return res_data