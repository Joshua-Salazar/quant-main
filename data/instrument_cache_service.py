from ..data.refdata import get_instruments_info


class InstrumentCacheService(dict):
    """
    InstrumentCacheService serves as an intermediary cache that holds instrument definitions by instrument id,
    It returns instrument definition from its cache if it can find it in its cache,
    It requests data from the cpcapdata server when it cannot and update its cache.
    """
    # TODO: This is only a very basic implementation
    # TODO: metaclass later
    # TODO: logger later
    # __metaclass__ = Singleton
    def __init__(self):
        dict.__init__(self)
        self._inst_ids = set()

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        s = '{} summary:\n'.format(self.__class__.__name__)
        row_fmt = '{:>30}: {:>12,d}\n'
        s += row_fmt.format('Total', len(self._inst_ids))
        for inst_id in self._inst_ids:
            s += str(self[inst_id]) + '\n'
        return s

    def __getitem__(self, inst_id):
        """
        Returns the definition for the requested instrument id. If the id is not in the cache, a request
        to the cpcapdata server will be made to populate the cache.
        :param inst_id: The instrument id
        :return: The instrument definition
        """
        if inst_id not in self._inst_ids:
            self._load_instrument(inst_id)
        return super(InstrumentCacheService, self).__getitem__(inst_id)

    def _load_instrument(self, inst_id):
        """
        Populates the cache with the instrument definition of the requested id.
        :param inst_id: The instrument id to download
        """
        inst_def_df = get_instruments_info([inst_id])
        if not len(inst_def_df):
            # TODO: logger later
            print('instrument definition of id : {} is not loaded from cpcapdata server.'.format(inst_id))
        elif len(inst_def_df) > 1:
            print('more than one instrument definition of id : {} are loaded from cpcapdata server.'.format(inst_id))
        else:
            inst_def = inst_def_df.to_dict('records')[0]
            self._inst_ids.add(inst_id)
            self[inst_id] = inst_def

    def load_instruments(self, inst_ids):
        for inst_id in inst_ids:
            self._load_instrument(inst_id)