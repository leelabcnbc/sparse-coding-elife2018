# then let's plot.
from matplotlib import pyplot as plt

def plot_all(final_result, save_predix=None, save=False):
    for dataset_this in ('A','B'):
        dict_this = final_result[dataset_this]
        plot_one(dict_this, {'A': 'Monkey A', 'B': 'Monkey B'}[dataset_this],
                 save_predix=save_predix, save=save)
#         plot_one(dict_this, {'NS_2250': 'Monkey A', 'NS_2250_MkB': 'Monkey B'}[dataset_this] + ', keep below',
#                 save=save, key_to_use=1)

def plot_one(dict_this, title, save_predix=None, save=False):
    
    style_list = [
        ({
             'marker': 'x',
             'color': 'r',
         },
         {
             'marker': 'x',
             'color': 'b',
         }),
        ({
             'marker': 'x',
             'color': 'r',
             'linewidth': 5,
             'alpha': 0.2,
         },
         {
             'marker': 'x',
             'color': 'b',
             'linewidth': 5,
             'alpha': 0.2,
         }),
        ({
             'marker': 's',
             'color': 'r',
             'linewidth': 1,
             'alpha': 0.8,
         },
         {
             'marker': 's',
             'color': 'b',
             'linewidth': 1,
             'alpha': 0.8,
         }),
        ({
             'marker': 'D',
             'color': 'r',
             'linewidth': 1,
             'alpha': 0.5,
         },
         {
             'marker': 'D',
             'color': 'b',
             'linewidth': 1,
             'alpha': 0.5,
         }),
        
        ({
             'marker': '*',
             'color': 'r',
             'linewidth': 1,
             'alpha': 0.5,
         },
         {
             'marker': '*',
             'color': 'b',
             'linewidth': 1,
             'alpha': 0.5,
         }),
    ]
    
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(title)
    # flip
    plt.xlim(0, 10)
    
    plt.xlabel('percentage of the top responses (%)')
    plt.ylim(0, 1)
    plt.ylabel('accuracy')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # plot line
    plt.axhline(y=dict_this['acc_default'], label='all', linestyle='--')
    
    # plot fills
    key_in_legend = {
        'zerofill': 'zero',
        'perneuron_median': "median on kept data, per neuron",
        'perneuron_median_overall': "median on all data, per neuron",
        'perneuron_nonparametric': 'nonp on kept data, per neuron, seed=0',
        'perneuron_nonparametric_seed1': 'nonp on kept data, per neuron, seed=1',
        'perneuron_nonparametric_overall': 'nonp on all data, per neuron',
        'perneuron_gaussian_noise': 'gaus on kept data, per neuron, seed=0',
        'perneuron_gaussian_noise_overall': 'gaus on all data, per neuron, seed=0',
        'zerofill_binary': 'zero, binary'
    }
    for idx_this, (key, fill_result) in enumerate(dict_this['fills'].items()):
        for key_to_use in [0,1]:
            style_this = style_list[idx_this][key_to_use]

            if key_to_use == 0:
                plt.plot(100-fill_result['thresh'],
                         fill_result['acc_above'], label=f'top only', **style_this)
            elif key_to_use == 1:
                plt.plot(100-fill_result['thresh'],
                         fill_result['acc_below'], label=f'top excluded', **style_this)
            else:
                raise RuntimeError('wrong!')
    # then at 0.5 draw a line.
    plt.axvline(x=0.5,linestyle='-', color='k',ymin=0,ymax=1,alpha=0.5, linewidth=0.5)
     
    plt.legend()
    
    # insert the 0-95 one
    
    ax_inset = plt.axes([.25, .6, .2, .2])
    ax_inset.axhline(y=dict_this['acc_default'], label='all', linestyle='--')
    for idx_this, (key, fill_result) in enumerate(dict_this['fills'].items()):
        for key_to_use in [0,1]:
            style_this = style_list[idx_this][key_to_use]

            if key_to_use == 0:
                plt.plot(100-fill_result['thresh'],
                         fill_result['acc_above'], label=f'keep above, fill below with {key}', **style_this)
            elif key_to_use == 1:
                plt.plot(100-fill_result['thresh'],
                         fill_result['acc_below'], label=f'keep below, fill above with {key}', **style_this)
            else:
                raise RuntimeError('wrong!')
    ax_inset.set_xlim(10, 100)
    ax_inset.set_xticks([10, 50,100])
    ax_inset.set_ylim(0,1)
    

    if save:
        assert save_predix is not None
        good_title = title.replace(' ', '_').replace(',', '_')
        plt.savefig(f'{save_predix}_{good_title}_final.png', dpi=300)
    # show after save. otherwise, somehow blank.
    plt.show()

    