import pickle
import numpy as np
import os

def save_obj(obj, name):
    with open('{}.pkl'.format(name),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('{}.pkl'.format(name), 'rb') as f:
        return pickle.load(f, encoding='latin1')

def preprocess_data(input_filename, output_dirname, 
                    sigs_0d, sigs_1d, sigs_predict,
                    lookback, delay=1, 
                    train_frac=.7, val_frac=.2,
                    separated=True, pad_1d_to=0,
                    save_data=True, noised_signal = None, sigma = 0.5, 
                    noised_signal_complete = None, sigma_complete = 1):
    
    # Gaussian normalization, return 0 if std is 0
    def normalize(obj, mean, std, maxs, mins):
        if False: 
            a=obj-mins
        else:
            a=obj-mean
        b=std
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    def finalize_signal(arr):
        arr[np.isnan(arr)]=0
        arr[np.isinf(arr)]=0
        return arr

    # load in the raw data
    data=load_obj(input_filename) #os.path.join(dirname,'final_data'))

    # extract all shots that are in the raw data so we can iterate over them
    shots = sorted(data.keys())    
    sigs = list(np.unique(sigs_0d+sigs_1d+sigs_predict))

    # first get the indices that contain all the data we need
    # (both train and validation)
    all_shots=[]
    for shot in shots:
       if set(sigs).issubset(data[shot].keys()):
           if all([data[shot][sig].size!=0 for sig in sigs]):
               all_shots.append(shot)
            
    data_all_times={}
    for sig in sigs+['time']:
        data_all_times[sig]=np.array([data[shot][sig] for shot in all_shots])
        data_all_times[sig]=np.concatenate(data_all_times[sig],axis=0)
        data_all_times[sig]=finalize_signal(data_all_times[sig])
    data_all_times['shot']=np.array([[shot]*data[shot][sigs[0]].shape[0] for shot in all_shots])
    data_all_times['shot']=np.concatenate(data_all_times['shot'],axis=0)

    indices={}
    subsets=['train','val']
    train_shots = all_shots[:int(len(all_shots)*train_frac)]    
    val_shots = all_shots[int(len(all_shots)*train_frac):int(len(all_shots)*(train_frac+val_frac))]
    subset_shots={'train':train_shots,'val':val_shots}
    
    def get_first_ind(arr,val):
        return np.searchsorted(arr,val) + lookback
    def get_last_ind(arr,val):
        return np.searchsorted(arr,val,side='right') - delay

    for subset in subsets:
        indices[subset]=[np.arange(get_first_ind(data_all_times['shot'],shot),get_last_ind(data_all_times['shot'],shot)+1) 
                         for shot in subset_shots[subset]]
        indices[subset]=np.concatenate(indices[subset])

    means={}
    stds={}
    mins={}
    maxs={}

    for sig in sigs:
        means[sig]=np.mean(data_all_times[sig][indices['train']],axis=0)
        stds[sig]=np.std(data_all_times[sig][indices['train']],axis=0)
        mins[sig]=np.amin(data_all_times[sig][indices['train']],axis=0)
        maxs[sig]=np.amax(data_all_times[sig][indices['train']],axis=0)

    data_all_times_normed={}
    for sig in sigs:
        data_all_times_normed[sig]=normalize(data_all_times[sig],means[sig],stds[sig],maxs[sig],mins[sig])

    target={}
    input_data={}
    times={} # never used
    for subset in subsets:
        final_target={}
        for sig in sigs_predict:
            final_target[sig]=data_all_times_normed[sig][indices[subset]+delay]-data_all_times_normed[sig][indices[subset]]
        #target[subset]=np.concatenate([final_target[sig] for sig in sigs_predict],axis=1)
        #target[subset] = np.array([final_target[sig] for sig in sigs_predict])
        target[subset] = final_target
        print("Sample Target shape for {} data: {}".format(subset, target[subset][sigs_predict[0]].shape))
        

        # alex's changes here
        pre_0d_dict = {}
        post_0d_dict = {}
        pre_1d_dict = {}
        for sig in sigs_0d:
            pre_0d_dict[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(-lookback,1)],axis=1)
            post_0d_dict[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(1,delay+1)],axis=1)
        for sig in sigs_1d:
            pre_1d_dict[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(-lookback,1)],axis=1)

        
        pre_input_0d = np.array([pre_0d_dict[sig] for sig in sigs_0d])

        pre_input_1d = np.array([pre_1d_dict[sig] for sig in sigs_1d])
    

        post_input_0d = np.array([post_0d_dict[sig] for sig in sigs_0d])
        
        
        print("Pre input 1d shape for {} data: {}".format(subset, pre_input_1d.shape))
        print("Pre input 0d shape for {} data: {}".format(subset, pre_input_0d.shape))
        print("Post input 0d shape for {} data: {}".format(subset, post_input_0d.shape))

        input_data[subset] = {"previous_actuators": pre_0d_dict, "previous_profiles": pre_1d_dict, "future_actuators": post_0d_dict}
        
        ########################
            


        # final_input={}
        # # only for if we want to append the 0d sigs to 1d sigs during the lookback steps (only applicable for separated)
        # final_input_appendage={}

        # final_input[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(delay+1)],axis=1)
        # for sig in sigs_0d:
        #     final_input_appendage[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(-lookback,1)],axis=1)
        #         for sig in sigs_1d:
        #         final_input[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(-lookback,1)],axis=1)
        # else:
        #     for sig in sigs_0d:
        #         final_input[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(-lookback,delay+1)],axis=1)
        #     for sig in sigs_1d:
        #         final_input[sig]=np.stack([data_all_times_normed[sig][indices[subset]+offset] for offset in range(-lookback,delay+1)],axis=1)

        # final_input_0d=np.concatenate([final_input[sig][:,:,np.newaxis] for sig in sigs_0d],axis=2)
        # final_input_1d=np.concatenate([final_input[sig] for sig in sigs_1d],axis=2)

        # if separated:
        #     input_data[subset]=[final_input_0d, final_input_1d]
        # else:
        #     final_input_1d[:,-delay:,:]=pad_1d_to
        #     input_data[subset]=np.concatenate([final_input_0d,final_input_1d],axis=2)            

    if save_data:
        print("Saving data to {}...".format(output_dirname))
        for subset in subsets:
            save_obj(data_all_times['time'][indices[subset]], os.path.join(output_dirname,'{}_time'.format(subset)))
            save_obj(data_all_times['shot'][indices[subset]], os.path.join(output_dirname,'{}_shot'.format(subset)))
            save_obj(target[subset],os.path.join(output_dirname,'{}_target'.format(subset)))
            save_obj(input_data[subset],os.path.join(output_dirname,'{}_data'.format(subset)))
        save_obj(means,os.path.join(output_dirname,'means'))
        save_obj(stds,os.path.join(output_dirname,'stds'))
        save_obj(means,os.path.join(output_dirname,'mins'))
        save_obj(means,os.path.join(output_dirname,'maxs'))

    else:
        return {'train_data': input_data['train'],
                'train_target': target['train'],
                'val_data': input_data['val'],
                'val_target': target['val'],
            }
