class Name_Ratio():
    """
    Class to store the confidence and ratio of a name
    (so that we dont recalculate it every time)
    """

    def __init__(self, name, confidence=None, ratio=None):
        self.name = name
        self.confidence = confidence
        self.ratio = ratio

    def __call__(self):
        return self.confidence, self.ratio


class Ratios():
    """
    Class to store the confidence and ratio of a name
    Will also remove the least used names if clean_up() is called
    """

    def __init__(self, df_ratios, names=[],
                 cut_off=KEEP_CUT_OFF_COUNT,
                 calc_if_found=CALC_IF_FOUND, max_word_len=MAX_WORD_LEN_DIFF, n_closest=N_CLOSEST):
        self.df_ratios = df_ratios
        # self.df_unique_list = set(self.df_ratios[NAME_DATA_COL].unique().tolist())
        self.df_inds = {name: index for index, name in
                        zip(self.df_ratios.index, self.df_ratios[NAME_DATA_COL].tolist())}
        self.names = dict()
        self.counts = dict()
        self.cutoff_count = cut_off
        self.calc_if_found = calc_if_found
        self.max_word_len = max_word_len
        self.n_closest = n_closest
        self.sel_cutoff = 0.001
        self.min_len = 2

        self.selections = dict()
        upper_lim = self.df_ratios[NAME_DATA_WORD_LEN_COL].max()
        for i in range(1, upper_lim):
            upper = min(i + self.max_word_len, upper_lim)
            lower = max(i - self.max_word_len, 0)

            self.selections[i] = self.df_ratios[
                (self.df_ratios[NAME_DATA_WORD_LEN_COL] >= lower) & (self.df_ratios[NAME_DATA_WORD_LEN_COL] <= upper)]

        if not isinstance(names, list):
            names = [names]

        for name in names:
            self._add_name(name, 1)


    # Returns the confidence and ratio of a name
    #  can take a string or a list of strings
    # @profile
    def __call__(self, names):
        should_ret_str = isinstance(names, str)
        if should_ret_str:
            names = [names]

        res_conf, res_ratio = [], []
        names = [internal_name_cleaning(name) for name in names]

        for name in names:
            conf, ratio = 0.0, 0.0
            if len(name) >= self.min_len:
                # if name not in self.names:
                self._add_name(name, 0)
                self.counts[name] += 1
                conf, ratio = self.names[name]()  # retrieves the confidence and ratio

            res_conf.append(conf)
            res_ratio.append(ratio)
        if should_ret_str:
            return res_conf[0], res_ratio[0]

        return res_conf, res_ratio

    # Calculate the confidence and ratio of a name based on averaging the most similar known names
    # @profile
    def process_name(self, name):
        n_closest_inds = None
        n_closest_dists = None
        n_closest_ratios = None
        n_closest_names = None

        df_ratios = self.df_ratios

        # if name in self.df_unique_list and not self.calc_if_found:
        if name in self.df_inds and not self.calc_if_found:

            n_closest_dists = [1.0]
            n_closest_ratios = [self.df_ratios.at[self.df_inds[name], NAME_DATA_RATIO_COL]]


        # Do we want to recalculate this even if it is already calculated?  The results could change as other values get added,
        #    but that means that a previous version of itself could influence current versions.
        #    Though this probably isnt an issue since this is function is called in _add_name, which checks if it is already in the list
        else:
            sel_ind = len(name)
            if sel_ind not in self.selections:
                # print('len name:', len(name))
                # print('sel:', list(self.selections.keys()))
                sel_ind = min(list(self.selections.keys()), key=lambda i: abs(i - sel_ind))
                # print(sel_ind)
            df_sel = self.selections[sel_ind]

            matchers = df_sel[NAME_SEQ_MATCHER_COL].to_numpy()
            _ = [f.set_seq1(name) for f in matchers]

            edit_dist = [f.quick_ratio() for f in matchers]
            n_closest_inds = np.argsort(edit_dist)[-self.n_closest * ACTUAL_CALC_MULT:]
            # print(edit_dist[n_closest_inds[0]], edit_dist[n_closest_inds[-1]])

            # print(-len(edit_dist) // ACTUAL_CALC_DENOM)
            # print(-self.n_closest * ACTUAL_CALC_MULT)

            edit_dist = [matchers[i].ratio() for i in n_closest_inds]

            n_closest_inds = np.argsort(edit_dist)
            n_closest_inds = n_closest_inds[-self.n_closest:] if len(
                n_closest_inds) > self.n_closest else n_closest_inds

            n_closest_dists = [edit_dist[i] for i in n_closest_inds]
            n_closest_ratios = df_sel.iloc[n_closest_inds][NAME_DATA_RATIO_COL].to_numpy()

        avg_confidence = np.average(n_closest_dists)
        avg_ratio = np.average(n_closest_ratios) if sum(n_closest_dists) == 0 else np.average(n_closest_ratios,
                                                                                              weights=n_closest_dists)
        # print(name, avg_confidence, n_closest_dists, avg_ratio, n_closest_ratios, n_closest_names)
        return avg_confidence, avg_ratio

    # Used to update the sequence matcher
    @staticmethod
    def matcher_update(matcher, name):
        matcher.set_seq1(name)
        return matcher.ratio()

    def _add_name(self, name, starting_count=1):
        name = internal_name_cleaning(name)
        if name not in self.names:
            conf, ratio = self.process_name(name)
            self.names[name] = Name_Ratio(name, conf, ratio)
            self.counts[name] = starting_count
            # print(self.names)

    # Frees up space by removing the names that are not as common
    def clean_up(self):
        if self.cutoff_count < len(self.names):
            to_keep = sorted(self.counts.values(), reverse=True)[:self.cutoff_count]
            min_to_keep = to_keep[-1]
            names_new = dict()
            kept_cnt = 0
            for name in self.names:
                if self.counts[name] >= min_to_keep:
                    names_new[name] = self.names[name]

                    kept_cnt += 1
                if kept_cnt >= self.cutoff_count:
                    break
            self.names = names_new
        self.counts = {name: 1 for name in self.names}


