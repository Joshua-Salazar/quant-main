
# Note, I would prefer to use UserDict, but with that, I have not yet got  the __setattr__ to work
class qdict(dict):
    """Dictionary wrapper, that als provide dot-access to dictionary keys".
       STILL EXPERIMENTAL -- DON'T USE FOR PRODUCTION
    """
    @staticmethod
    def from_dict(d: dict):
        rv = qdict()
        for key, value in d.items():
            if isinstance(value, dict):
                rv[key] = qdict.from_dict(value)
            elif isinstance(value, list):
                rv[key] = [qdict.from_dict(x) for x in value]
            else:
                rv[key] = value
        return rv

    def __init__(self, args=None):
        if args is not None:
            super().__init__(qdict.from_dict(args))
        else:
            super().__init__()

    def __dir__(self):
        return list(self.keys()) + super().__dir__()

    def as_dict(self):
        raise NotImplementedError("as_dict not implemented")

    def __setattr__(self, name, value):
        super().update({name: value})

    def __delattr__(self, attr):
        if attr in self.keys():
            super().pop(attr, None)
        else:
            super().__delattr__(attr)

    def __getattr__(self, attr):
        if attr in self.keys():
            return super().get(attr)
        else:
            super().__getattr__(attr)

def test1():
    # INDEX-OPERATOR ASSIGNMENT
    q1 = qdict()
    q1['monday'] = 1
    q1['tuesday'] = 2
    q1['week'] = qdict({'jan': 31, 'feb': 28})
    q1['year'] = dict({'2021': True, '2022': False})

    q1 = qdict.from_dict({'monday': 1,
     'tuesday': 2,
     'week': {'jan': 31, 'feb': 28},
     'year': {'2021': True, '2022': False}})

    # EQUIVALENT ASSIGNMENT

    # write (one levels)
    assert q1['monday'] == 1
    assert q1.monday == 1

    q1['monday'] = 0
    assert q1['monday'] == 0
    assert q1.monday == 0

    q1.monday = -1
    assert q1['monday'] == -1
    assert q1.monday == -1

    # write (two levels)
    assert q1['week']['jan'] == 31
    assert q1.week.jan == 31

    q1['week']['jan'] = 30
    assert q1['week']['jan'] == 30
    assert q1.week.jan == 30

    q1.week.jan = 29
    assert q1['week']['jan'] == 29
    assert q1.week.jan == 29

    # KEY LOOKUP
    assert ('monday' in q1)
    assert ~('x' in q1)

    # TO STRING
    #print("{}".format(q1))

    # KEYS
    list(q1.keys()) == ['monday', 'tuesday', 'week', 'year']

    # DELETE VIA COMPLETION
    del q1.monday
    del q1['year']
    list(q1.keys()) == ['tuesday', 'week']


def test2():
    dd = {
        'env': 'uat',
        'oil':  {'CLZ2': 1.2, 'CLZ3': 1.3},
        'index': {"ES": {"Z2": 2.0, "Z3": 3.0}},
        'pos': [{"CLZ2": 100}, {"ESZ2": 200}] }

    qq = qdict({
        'env': 'uat',
        'oil':  {'CLZ2': 1.2, 'CLZ3': 1.3},
        'index': {"ES": {"Z2": 2.0, "Z3": 3.0}},
        'pos': [{"CLZ2": 100}, {"ESZ2": 200}]})

    assert dd == qq
    assert f"{dd}" == f"{qq}"

    # add new item, traditional
    for x in dd, qq:
        x["index"]["ES"]["Z4"] = 4.0
    assert dd == qq
    assert f"{dd}" == f"{qq}"

    # add new item, dot-access
    dd["index"]["ES"]["Z5"] = 5.0
    qq.index.ES.Z5 = 5.0
    assert dd == qq
    assert f"{dd}" == f"{qq}"

    # deletion
    del dd["index"]["ES"]["Z5"]
    del qq.index.ES.Z5
    assert dd == qq
    assert f"{dd}" == f"{qq}"

    print(qq)
    print(dd)


if __name__ == "__main__":
    test1()
    test2()
