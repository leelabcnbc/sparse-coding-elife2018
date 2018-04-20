from itertools import product
from collections import OrderedDict
import numpy as np
import h5py
from decoder import classification_wrapper_one_Xy, accuracy_score


def threshold_data(data, method, threshold, keep_above=True,
                  data_to_use=None):
    # this is a rewrite of `threshold_neural_data`
    # from <https://github.com/leelabcnbc/sparse-coding-tang/blob/master/sct_code_python/preprocessing.py>
    if data_to_use is None:
        data_to_use = data
    
    assert np.all(np.isfinite(data))
    assert np.all(np.isfinite(data_to_use))
    assert data_to_use.shape == data.shape
    if method == 'raw':
        # this is method used in legacy.
        mask = data <= threshold
    elif method == 'percentile':
        mask = data <= np.percentile(data.ravel(), threshold)
    else:
        raise ValueError(f'unsupported method `{method}`')

    if not keep_above:
        mask = np.logical_not(mask)

    data_all = data_to_use.copy()
    data_all[mask] = np.nan
    return data_all, mask


def get_flat_data_label_and_group(data_all):
    n_stim, max_trial, n_neuron = data_all.shape
    # make the labels and group in shape (n_stim, max_trial)
    # this is a rewrite of `get_n_trial_tang_data`
    # from <https://github.com/leelabcnbc/sparse-coding-tang/blob/master/sct_code_python/preprocessing.py>

    y_label = np.tile(np.arange(n_stim), max_trial)
    y_group = np.repeat(np.arange(max_trial), n_stim)
    # this follows the old way reshaping is done. essentially, first have all 1st trial data,
    # then all 2nd trial, then all 3rd trial, etc.
    x_flat = np.concatenate([data_all[:, i_trial, :] for i_trial in range(max_trial)], axis=0)

    return x_flat, y_label, y_group


def fill_in_data(data_all_original: np.ndarray, data_mean):
    # this will fill in data_all_original's all NaN with data_mean, in corresponding locations.
    data_all_original = data_all_original.copy()
    assert np.isscalar(data_mean)
    data_all_original[np.isnan(data_all_original)] = data_mean
    return data_all_original

def get_one_neuron_data_all(dataset):
    with h5py.File(f'data_{dataset}.hdf5', 'r') as f:
        X = f[dataset]['all'][:,:3]
        X_mean = f[dataset]['mean'][...]
    return {
        'X': fill_in_data(X, 0),
        'X_mean': X_mean,
    }
    
def get_data():
    dataset_list = ('A',
                    'B',)
    data_dict_ = OrderedDict()
    for dataset in dataset_list:
        data_dict_[dataset] = get_one_neuron_data_all(dataset) 
    
    # make all being zero.
    return data_dict_
data_dict = get_data()

# then get the labels.
 
data_collect_dict_ = OrderedDict()
    
dataset_list = list(data_dict.keys())


param_list = OrderedDict(
    NearestCentroid=('NearestCentroid', {}, None),
)


fill_scheme_list = (
    None,
    'zerofill',
)

thresh_scheme_list = ('no_thresh',
                      'thresh_percentile',
                     )


_threshold_scheme_dict = {
    'no_thresh': None,  # None just mean raw stuff. this is special.
    'thresh_percentile': 'percentile'
}


percentile_list_global = (1,
                          10, 20, 30, 40, 50, 60, 70, 80, 90,
                          91, 92, 93, 94, 95, 96, 97, 98, 99,
                          99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9)

def generate_save_group(dataset_this, param_this, fill_scheme_this, thresh_scheme_this, percentile_this):
    if percentile_this is None:
        assert fill_scheme_this is None and thresh_scheme_this == 'no_thresh'
        grp_to_save = f'{dataset_this}/{param_this}/{fill_scheme_this}/{thresh_scheme_this}'
    else:
        assert fill_scheme_this is not None and thresh_scheme_this != 'no_thresh'
        grp_to_save = f'{dataset_this}/{param_this}/{fill_scheme_this}/{thresh_scheme_this}' + '/{:.2f}'.format(percentile_this)
    return grp_to_save

def collect_all_data(cache_file, normalize):
    with h5py.File(cache_file) as f_out:
        for dataset_this, param_this, fill_scheme_this, thresh_scheme_this in product(
            dataset_list, param_list, fill_scheme_list, thresh_scheme_list
        ):
            # remove incompatible cases
            if fill_scheme_this is None and thresh_scheme_this != 'no_thresh':
                continue
            if fill_scheme_this is not None and thresh_scheme_this == 'no_thresh':
                continue
                
            if fill_scheme_this is None and thresh_scheme_this == 'no_thresh':
                trivial_flag = True
            else:
                assert fill_scheme_this is not None and thresh_scheme_this != 'no_thresh'
                trivial_flag = False
                
            if trivial_flag:
                percentile_list = (None,)
            else:
                percentile_list = percentile_list_global
            data_all=data_dict[dataset_this]['X']
            data_all_mean=data_dict[dataset_this]['X_mean']
            
            # check two max.
            if normalize:
                max1 = np.max(data_all_mean, axis=0)
                max2 = np.max(data_all, axis=(0,1))
                assert np.all(max1>0)
                assert np.all(max2>0)
                assert max1.shape == max2.shape
                print(max1.shape)
                print((max2/max1).max(), (max2/max1).min(),
                      (max2/max1).mean(), np.median((max2/max1)))
                # then use data_all_mean's max to normalize data.
                # always use normalized data.
                #data_all_mean_max = np.max(data_all_mean, axis=0)

                data_all_mean_max = max1

                data_all_old = data_all
                data_all = data_all/data_all_mean_max
                data_all_mean_old = data_all_mean
                data_all_mean = data_all_mean/data_all_mean_max
            
                assert not np.may_share_memory(data_all_mean, data_all_mean_old)
                assert not np.may_share_memory(data_all, data_all_old)
            
            assert np.all(np.isfinite(data_all_mean)) and np.all(np.isfinite(data_all))
                
            x_flat, y_label, y_group = get_flat_data_label_and_group(data_all=data_all)
            x_flat_old = get_flat_data_label_and_group(data_all=data_all_old)[0] if normalize else None       
            
            print('data stat mean {:.2f}, std {:.2f}, min {:.2f}, max {:.2f}'.format(x_flat.mean(), x_flat.std(), x_flat.min(), x_flat.max()))
            for percentile_this in percentile_list:
                print(dataset_this, param_this, fill_scheme_this, thresh_scheme_this, percentile_this)
                a = save_one_case_simple(dataset_this, param_this, fill_scheme_this, thresh_scheme_this, percentile_this, f_out, x_flat, y_label, y_group, extra_data_3=x_flat_old)
                data_collect_dict_[(dataset_this, param_this, fill_scheme_this, thresh_scheme_this, percentile_this)] = a
    return data_collect_dict_

def save_one_case_simple(dataset_this, param_this, fill_scheme_this, thresh_scheme_this, percentile_this, f_out, x_flat,
                         y_label, y_group, extra_data_3=None):
    grp_to_save_this = generate_save_group(dataset_this, param_this, fill_scheme_this, thresh_scheme_this, percentile_this)
    thresh_data = _threshold_scheme_dict[thresh_scheme_this]
    if grp_to_save_this not in f_out:
        # then compute
        if thresh_data is None:
            assert fill_scheme_this is None
            result_dict = save_one_case_default(extra_data_3 if extra_data_3 is not None else x_flat, y_label, y_group, param_list[param_this])
        else:
            assert thresh_data == 'percentile'
            assert fill_scheme_this is not None
            result_dict = save_one_case_percentile(x_flat,y_label,y_group,param_list[param_this],percentile_this,fill_scheme_this, extra_data_3=extra_data_3)
        # then save
        save_to_hdf5(result_dict, grp_to_save_this, f_out)
    result_dict = load_from_hdf5(grp_to_save_this, f_out)
    
    return result_dict

def save_one_case_default(x_flat, y_label, y_group, params):
    
    y_pred = classification_wrapper_one_Xy(x_flat,
                                           y_label, params, None,
                                           y_group, -1)
    result_dict = OrderedDict(
        [('y_original', y_label), ('y_pred', y_pred)],
    )
    return result_dict

def save_one_case_percentile(x_flat, y_label, y_group, params, percentile_this, fill_scheme_this, extra_data_3=None):
    X_above, _ = threshold_data(x_flat, 'percentile', percentile_this, data_to_use=extra_data_3)
    X_below, mask = threshold_data(x_flat, 'percentile', percentile_this, keep_above=False, data_to_use=extra_data_3)
    
    assert fill_scheme_this == 'zerofill'
    X_above = fill_in_data(X_above, 0)
    X_below = fill_in_data(X_below, 0)
    
    y_pred_above_this = classification_wrapper_one_Xy(X_above, y_label, params, None, y_group, -1)
    y_pred_below_this = classification_wrapper_one_Xy(X_below, y_label, params, None, y_group, -1)
    result_dict = OrderedDict(
        [('y_original', y_label), ('thresh', percentile_this),
         ('mask_portion_above', mask.mean()),
         ('y_pred_above', y_pred_above_this),
         ('y_pred_below', y_pred_below_this),
        ]
    )
    return result_dict

def get_keys_to_save(grp_to_save_this):
    key_parts = grp_to_save_this.split('/')
    
    
    if len(key_parts) == 5:
        keys_to_save = ('y_original', 'thresh', 'mask_portion_above', 'y_pred_above', 'y_pred_below')
    elif len(key_parts) == 4:
        keys_to_save = ('y_original', 'y_pred')
    else:
        raise RuntimeError('impossible!')
        
    return keys_to_save

def save_to_hdf5(result_dict, grp_to_save_this, f_out):
    assert grp_to_save_this not in f_out
    grp_this = f_out.create_group(grp_to_save_this)
    
    
    keys_to_save = get_keys_to_save(grp_to_save_this)
    assert result_dict.keys() >= set(keys_to_save)
    for key in keys_to_save:
        grp_this.create_dataset(key, data=result_dict[key])
        # otherwise it will be cached, which can be unintuitive.
        f_out.flush()
        
def load_from_hdf5(grp_to_save_this, f_out):
    grp_this = f_out[grp_to_save_this]
    keys_to_restore = get_keys_to_save(grp_to_save_this)
    # then load data.
    data = OrderedDict()
    for key in keys_to_restore:
        data[key] = grp_this[key][...]
    return data


def collect_all_data_final(data_collect_dict):
#     for dataset_this, param_this, fill_scheme_this, thresh_scheme_this in product(
#             dataset_list, param_list, fill_scheme_list, thresh_scheme_list
#         ):
    final_result = OrderedDict()
    for dataset_this in dataset_list:
        result_this_datset = OrderedDict()
        # first, collect default.
        result_default = data_collect_dict[dataset_this, 'NearestCentroid', None, 'no_thresh', None]
        acc_default = accuracy_score(result_default['y_original'], result_default['y_pred'])
        
        result_this_datset['acc_default'] = acc_default
        result_this_datset['fills'] = OrderedDict()
        # then for each mode of fill
        for fill_scheme in [x for x in fill_scheme_list if x is not None]:
            # for loop to collect all data
            thresh_this = []
            above_this = []
            below_this = []
            for percentile_this in percentile_list_global:
                result_this = data_collect_dict[dataset_this, 'NearestCentroid', fill_scheme, 'thresh_percentile', percentile_this]
                thresh_this.append(percentile_this)
                assert result_this['thresh'][()] == percentile_this
                above_this.append(accuracy_score(result_this['y_original'],
                                                 result_this['y_pred_above']))
                below_this.append(accuracy_score(result_this['y_original'],
                                                 result_this['y_pred_below']))
            # then append the 100 one
            thresh_this.append(100)
            # chance.
            above_this.append(1/2250)
            below_this.append(acc_default)
            result_this_datset['fills'][fill_scheme] = {
                'thresh': np.asarray(thresh_this),
                'acc_above': np.asarray(above_this),
                'acc_below': np.asarray(below_this),
            }
        final_result[dataset_this] = result_this_datset
    return final_result