import cobra
from matplotlib import pyplot as plt
from util2 import chemostat, batch, fed_batch, batch_multipleC, batch_multipleC_reg, batch_o2lim
import numpy as np
import pickle

def get_ids(model):
    for r in model.reactions:
        if 'oxygen exchange' in r.name:
            oxygen_id = r.id
        if 'glucose exchange' in r.name:
            glc_exc_id = r.id
        if 'growth' in r.name and not 'non' in r.name:
            growth_id = r.id
    return oxygen_id, glc_exc_id, growth_id
        
def feed(t):
    cs_in = 209/198*1e3 #mmol/kg
    # Batch lasts 4h, then exponential feed:
    t_start = 4
    if t <= t_start:
        F = 0
    else:
        F = 6e-5*np.exp(0.0636*(t-t_start)) #kg/h
    return(F, cs_in)

def data_generator(model_path, list_reactor_type, dilution_rate, list_input_concentrations, multiple_substrate_conc, vol_batch, batch_con, fed_batch_con, gecko=False, volume_cstr=1, initial_batch = 8):
    model = cobra.io.read_sbml_model(model_path)
    
    data_dic = {}
    
    o2_exchange_id, glc_exchange_id, growth_id = get_ids(model)
        
    for reactor_type in list_reactor_type:
        data_dic[reactor_type] = {}
        if reactor_type=='cstr':
            for cstr_con in list_input_concentrations:
                data_dic[reactor_type][cstr_con] = {}
        
                flux_y8, con_y8 = chemostat(model, gecko, growth_id, glc_exchange_id, o2_exchange_id, o2_id = 's_1277__91__e__93__', co2_id = 's_0458__91__e__93__', D_list = np.arange(0.05, dilution_rate, 0.02).tolist(), V = volume_cstr, cstr_con = cstr_con/0.18, F_g = 0.5*60, c_ogin = 0.21*1000/(0.082 * (21+273)), printing = False )
                data_dic[reactor_type][cstr_con]['flux'] = flux_y8
                data_dic[reactor_type][cstr_con]['conc'] = con_y8
            
        if reactor_type == 'batch' or reactor_type=='fed batch' or 'batch multiple sources' or 'batch multiple reg':
            if gecko==True:
                prot_ub = model.reactions.prot_pool_exchange.upper_bound
                model.reactions.prot_pool_exchange.upper_bound = 1.25 * prot_ub
                model.reactions.r_1761.upper_bound = 100
            
            if reactor_type=='batch':
                cs_0 = batch_con/0.18
                data_dic[reactor_type][batch_con] = {}
                
                dt, cx_0 = 0.5, 0.1
                
                flux_y8, con_y8 = batch(model, gecko, cs_0, cx_0, vol_batch, dt, glc_exchange_id, growth_id, qs_max=10, kms=0.28, t_end=15)
                con_y8 = con_y8.set_index('t')
                data_dic[reactor_type][batch_con]['flux'] = flux_y8
                data_dic[reactor_type][batch_con]['conc'] = con_y8
        
            if reactor_type == 'fed batch':
                cs_0 = fed_batch_con/198*1e3
                cx_0=0.275
                V_0=0.365
                t_end=124
                dt = 0.5
                t_lag=0.5
                qs_max=10
                kms = 0.28
                data_dic[reactor_type][fed_batch_con] = {}
                
                flux_y8, con_y8, v_y8 = fed_batch(model, cs_0, cx_0, V_0, feed, t_end, t_lag, dt, gecko, glc_exchange_id,growth_id, qs_max, kms)
                con_y8 = con_y8.set_index('t')
                data_dic[reactor_type][fed_batch_con]['flux'] = flux_y8
                data_dic[reactor_type][fed_batch_con]['conc'] = con_y8
                
            if reactor_type == 'batch multiple sources' or 'batch multiple reg':
                
                dataset_copy = multiple_substrate_conc.copy()
                values =[]
                
                for i in dataset_copy:# sets all the substrate values to 0
                    values.append(multiple_substrate_conc[i][1])
                    dataset_copy[i][1] = 0
                
                for i in range(len(multiple_substrate_conc)-1):
                    for j in range(i+1, len(multiple_substrate_conc)):
                        a = values[i]
                        b = values[j]
                        
                        substrates = list(multiple_substrate_conc.keys())[i] + ' and ' + list(multiple_substrate_conc.keys())[j]
                        
                        dataset_copy[list(multiple_substrate_conc.keys())[i]][1] = a
                        dataset_copy[list(multiple_substrate_conc.keys())[j]][1] = b
                        
                        data_dic[reactor_type][substrates] = {}
                        
                        cx_0, V, dt, t_end = 1e-3, 3.5, 0.5, 25
                        
                        if reactor_type == 'batch multiple sources':
                            flux_y8, con_y8 = batch_multipleC(model, gecko, growth_id, dataset_copy, cx_0, V, dt, t_end)
                        else:
                            if gecko:
                                flux_y8, con_y8 = batch_multipleC_reg(model, growth_id, dataset_copy, cx_0, V, dt, t_end)
                            else:
                                break
                        con_y8 = con_y8.set_index('t')
                        
                        data_dic[reactor_type][substrates]['flux'] = flux_y8
                        data_dic[reactor_type][substrates]['conc'] = con_y8
                        
                        dataset_copy[list(multiple_substrate_conc.keys())[i]][1] = 0
                        dataset_copy[list(multiple_substrate_conc.keys())[j]][1] = 0
                        
        if reactor_type == 'lactate producing':
            cs_0, cx_0, V, dt, qs_max, kms, t_end = 100/0.18, 0.05, 1, 0.5, 10, 0.28, 180
            data_dic[reactor_type][cs_0*0.18] = {}

            flux_y8, con_y8 = batch_o2lim(model, gecko, cs_0, cx_0, V, dt, glc_exchange_id, growth_id, qs_max, kms, t_end)
            con_y8 = con_y8.set_index('t')
            data_dic[reactor_type][cs_0*0.18]['flux'] = flux_y8
            data_dic[reactor_type][cs_0*0.18]['conc'] = con_y8

    with open('value_dictionary.pkl', 'wb') as f:
        pickle.dump(data_dic, f)
    
    return data_dic

list_reactor_type = ['cstr', 'batch', 'fed batch', 'batch multiple sources', 'batch multiple reg','lactate producing']  # could be batch, cstr, fed batch, batch multiple sources, batch multiple reg, lactate producing
list_input_concentration = [30, 15, 7.5, 10]
batch_con = 8
fed_batch_con = 2.5
vol_batch = 0.215
multiple_substrate_conc = {    
    'D-glucose [extracellular]':['r_1714', 20/180e-3],
    'sucrose [extracellular]':['r_2058', 15/342e-3],
    'D-fructose [extracellular]': ['r_1709', 20/180.156e-3],
    'D-mannose [extracellular]': ['r_1715', 17/180.156e-3]
}

model_path = 'E:\lectures\Project\ecmodels-predict-growth-dynamics-s.-cerevisiae-main-Models\Models\yeastGEM.xml'


x = data_generator(model_path, list_reactor_type, 0.4, list_input_concentration, multiple_substrate_conc, batch_con, fed_batch_con, vol_batch)



def plot(x, list_reactor_type, conc, parameter_dic):
    for reactor_type in list_reactor_type:
        for i in parameter_dic[reactor_type]:
            plotting = i
            for j in conc:
                x[reactor_type][j]['conc'][plotting].plot()
                plt.ylabel(j)
                plt.legend()
            plt.show()
        
    # return

parameter_dic = {}
parameter_dic['cstr'] = ['biomass [cytoplasm]']

plot(x, list_reactor_type, list_input_concentration, parameter_dic)