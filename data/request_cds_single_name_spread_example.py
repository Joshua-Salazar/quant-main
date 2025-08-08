import requests

"""
    For single name CDS, the quotes are coming from Automarks.
    Automarks will get the quotes from different sources from BBG and aggregates them.
    Automarks will save the quotes in data_cache_otc.
    You need to know the CTP ID for the CDS you are interested in.
"""

# Let's take "ASSGEN CDS EUR SR 7Y" as example
# "ASSGEN CDS EUR SR 7Y" id in CTP is 12453762

# this is coming from data_cache_otc
url = f'http://cpiceregistry:6703/marks?server_name=data_cache_otc&instId={12453762}'

results = requests.get(url)

data = results.json()

for d in data:
    if 'key' in d:
        key = d['key']
        if 'inst_id' in key and 'pricing_source' in key:
            print(f"Data for {key['inst_id']} from source {key['pricing_source']}\n")

    if 'current_mark' in d:
        current_marks = d['current_mark']
    else:
        current_marks = dict()

    for unit, value in current_marks.items():
        print(f"Quote value is {value} in {unit}.\n")

    if d['is_override']:
        print("It is an override.")

    if 'broker_quote_comments' in d:
        broker_quote_comments = d['broker_quote_comments']
    else:
        broker_quote_comments = dict()

    if len(broker_quote_comments) > 0:
        print('Brokers:')

    for broker, comment in broker_quote_comments.items():
        print(f"{broker.split('|')[1]}: {comment}")

print('ok')
