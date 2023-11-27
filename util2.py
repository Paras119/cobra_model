#!/usr/bin/env python
# coding: utf-8
# author: Sara Moreno-Paz (saramorenopaz@gmail.com)

import numpy as np
import pandas as pd
import cobra
# import cobra.test
from cobra import Model, Reaction, Metabolite
import warnings
warnings.filterwarnings("error", category = UserWarning)

# --------------------------------------------------------
# CHEMOSTAT FUNCTIONS
# --------------------------------------------------------

def chemostat_sim(model, gecko, growth_id, glc_exchange_id, o2_exchange_id, o2_id, co2_id, D, V, c_sin, 
                  F_g, c_ogin, printing):
    
    '''
    Simulates a given GEM under chosen chemostat operational conditions.
    Gives fluxes through exchange reactions in mmol/gDW/h (or h-1 for growth) and concentrations:
        - Biomass concentration in the reactor: gDW/L
        - Substrates and products concentrations in the reactor: mmol/L
        - Oxygen and CO2 concentrations in gas out: mmol/Lgas
    Inputs: 
        - model: GEM
        - gecko: True if ecGEM, False otherwise
        - growth_id: id of biomass production reaction
        - glc_exchange_id: id of glucose exchange reaction (REV if gecko)
        - o2_exchange_id: id of oxygen exchange reaction (REV if gecko)
        - o2_id: metabolite id for oxygen
        - co2_id: metabolite id for CO2
        - D: dilution rate of the reactor in h-1
        - c_sin: glucose concentration in the feed in mmol/l
        - F_g: gas flow rate in L/h
        - c_ogin: oxygen concentration in the gas in mmol/l
        - printing: if True prints calculated concentrations
       ''' 
    
    # Fix growth rate to dilution rate (+- 10%)
    model.reactions.get_by_id(growth_id).bounds = 0.9*D, 1.1*D
    
    # Unconstrain glucose uptake
    model.reactions.get_by_id(glc_exchange_id).bounds = -1000, 1000
    # Min glc uptake
    model.objective = glc_exchange_id
    if gecko == False:
        model.objective.direction = 'max' # Uptake is negative
    else: 
        model.objective.direction = 'min' # Uptake is positive
    
    # Perform FBA
    try:
        solution = model.optimize()
    except (UserWarning) as e:
        return(pd.DataFrame.from_dict({'Infeasible': [True]}), pd.DataFrame.from_dict({'Infeasible': [True]}))
    
    # Find active exchange reactions 
    fluxes = {}
    for reaction in model.reactions:
        reaction_id = reaction.id
        if len(model.reactions.get_by_id(reaction_id).metabolites) == 1: #Only exchange reactions
            flux = solution.fluxes[reaction_id]
            if abs(flux) > 1e-4:
                fluxes[reaction_id] = flux
    
    # Calculate cx
    cx = D*c_sin/abs(fluxes[glc_exchange_id])
    
    # Calculate concentration of by-products
    concentrations = {}
    exceptions = [o2_exchange_id, glc_exchange_id]
        # Get glc id
    for metabolite in model.reactions.get_by_id(glc_exchange_id).metabolites:
        glc_id = metabolite
    
    if gecko == False: # only products have positive fluxes through exchange reactions
        for reaction, flux in fluxes.items():
            if flux > 0 or model.reactions.get_by_id(reaction).id in exceptions:
                metabolites = model.reactions.get_by_id(reaction).metabolites
                for metabolite in metabolites: 
                    if metabolite.id == o2_id: 
                        concentrations[metabolite.id] = (F_g*c_ogin - abs(flux)*cx*V)/F_g
                    elif metabolite.id == co2_id:
                        concentrations[metabolite.id] = (flux*cx*V)/F_g
                    elif metabolite == glc_id:
                        concentrations[metabolite.id] = (D*c_sin - abs(flux)*cx)/D
                    else:                       
                        concentrations[metabolite.id] = flux * cx/D
    else:
        for reaction, flux in fluxes.items():# product exchange reactions (excretion) don´t have REV in their names
            if 'REV' not in reaction and 'prot_pool_exchange' not in reaction:
                metabolites = model.reactions.get_by_id(reaction).metabolites
                for metabolite in metabolites:
                    if metabolite.id == co2_id:
                        concentrations[metabolite.id] = (flux*cx*V)/F_g
                    else:                       
                        concentrations[metabolite.id] = flux * cx/D
            if model.reactions.get_by_id(reaction).id in exceptions:
                metabolites = model.reactions.get_by_id(reaction).metabolites
                for metabolite in metabolites: #metabolites is a dict with metabolite as key and coefficeint as value
                    if metabolite.id == o2_id:
                        concentrations[metabolite.id] = (F_g*c_ogin - abs(flux)*cx*V)/F_g
                    elif metabolite == glc_id:
                        concentrations[metabolite.id] = (D*c_sin - abs(flux)*cx)/D
    
    # Prepare for printing and return
    l_fluxes = []
    l_conc = []
    df_fluxes = pd.DataFrame()
    df_conc = pd.DataFrame()
    
    for r, f in fluxes.items():
        l_fluxes.append([r, model.reactions.get_by_id(r).name, f])
        df_fluxes[r] = [f]
    df_fluxes_print = pd.DataFrame.from_records(l_fluxes, columns = ['Reaction_id', 'Reaction name', 'Flux'])
    
    for m, c in concentrations.items():
        l_conc.append([m, model.metabolites.get_by_id(m).name, c])
        df_conc[model.metabolites.get_by_id(m).name] = [c] #get metabolite name
    df_conc_print = pd.DataFrame.from_records(l_conc, columns = ['Metabolite_id', 'Metabolite name', 'Concentration'])
    
    #  Printing
    if printing == True:
        print(df_fluxes_print, '\n \n', df_conc_print)
            
    # Return two data frames in which column names are reaction or metabolite id and values are fluxes or concen
    return(df_fluxes, df_conc)


def chemostat(model, gecko, growth_id, glc_exchange_id, o2_exchange_id, o2_id, co2_id, D_list, V, c_sin, 
                  F_g, c_ogin, printing):
    '''
    Simulates a given GEM under chosen chemostat operational conditions at a range of dilution rates (D).
    Gives fluxes through exchange reactions in mmol/gDW/h (or h-1 for growth) and concentrations per D:
        - Biomass concentration in the reactor: gDW/L
        - Substrates and products concentrations in the reactor: mmol/L
        - Oxygen and CO2 concentrations in gas out: mmol/Lgas
    Inputs: 
        - model: GEM
        - gecko: True if ecGEM, False otherwise
        - growth_id: id of biomass production reaction
        - glc_exchange_id: id of glucose exchange reaction (REV if gecko)
        - o2_exchange_id: id of oxygen exchange reaction (REV if gecko)
        - o2_id: metabolite id for oxygen
        - co2_id: metabolite id for CO2
        - D_list: list of desired dilution rates for simulations (in h-1)
        - c_sin: glucose concentration in the feed in mmol/l
        - F_g: gas flow rate in L/h
        - c_ogin: oxygen concentration in the gas in mmol/l
        - printing: if True prints calculated concentrations
       ''' 
    for D in D_list:
        flux, con = chemostat_sim(
            model, gecko, growth_id,
            glc_exchange_id, o2_exchange_id,
            o2_id, co2_id,
            D, V, c_sin, F_g, 
            c_ogin, printing
        )
        # Create output df
        if D == D_list[0]: 
            df_con = con
            df_flux = flux
        else: 
            df_con = df_con.append(con, sort = 'False') 
            df_flux = df_flux.append(flux, sort = 'False')
    df_con['D'] = D_list
    df_con = df_con.set_index('D')
    df_flux['D'] = D_list
    df_flux = df_flux.set_index('D')
    return(df_flux, df_con)

# --------------------------------------------------------
# BATCH FUNCTIONS 
# --------------------------------------------------------

def batch(model, gecko, cs_0, cx_0, V, dt, glc_exc_id, growth_id, qs_max, kms, t_end):
    '''
    Simulate a given GEM in a batch reactor assuming glc limitation, calculates
    metabolite concentrations in the reactor using Euler´s integration method.
    Ethanol uptake is allowed if ethanol is present in the reactor at a max
    uptake rate of 5 mmol/gDW/h
    Inputs: 
        - model: GEM
        - gecko: True if ecGEM, False otherwise
        - c_s0: initial glucose concentration in the reactor (mmol/l)
        - c_x0: initial biomass concentration in the reactor (gDW/l)
        - V: volume of the reactor (L)
        - glc_exchange_id: id of glucose exchange reaction (REV if gecko)
        - growth_id: id of biomass production reaction
        - qs_max: maximum glucose uptake rate in mmol/gDW/h
        - ks: Michaeilis-Menten constant for glucose uptake in mmol/l
        - t_end: duration of the batch (h)
    Outputs:
        - all_flux: dataframe with fluxes through exchange reactions at each t
        - all_con: dataframe with metabolite concentrations in the reactor at each t        
    '''
    
    # Create dataframe to store data
    all_con = pd.DataFrame({'t': [0],
                            'glucose [extracellular]': [cs_0],
                            'biomass [cytoplasm]': [cx_0],
                            'ethanol [extracellular]': [0]
                              })
    all_flux = pd.DataFrame({'t': [0]})
    cs = all_con['glucose [extracellular]'].iloc[-1]
    t = 0
    while t < t_end:
        t = t + dt
        df_flux = pd.DataFrame()
        df_con = pd.DataFrame()
        
        # Calculate qs and set as lower/upper bound             
        qs = qs_max * cs/(kms+cs)
        if gecko == False:
            model.reactions.get_by_id(glc_exc_id).lower_bound = -qs
        else:
            if qs < 0:
                qs = 0
            model.reactions.get_by_id(glc_exc_id).upper_bound = qs
            
        # ETH uptake
        E = all_con['ethanol [extracellular]'].iloc[-1]
        if gecko == True:
            if E > 0:
                model.reactions.r_1761_REV.upper_bound = 5
            else:
                model.reactions.r_1761_REV.upper_bound = 0

        # Perform FBA
        model.objective = growth_id
        model.objective.direction = 'max'
        try:
            solution = model.optimize()
        except (UserWarning) as e:
            print('Solution infeasible, t =', t)
            break
        
        # Check cs
        Ms = (
            all_con['glucose [extracellular]'].iloc[-1] * V 
            - dt * abs(solution.fluxes[glc_exc_id]) 
            * all_con['biomass [cytoplasm]'].iloc[-1] * V
        )
        
        cs = Ms/V
        if cs > 0:
            cs = cs
        else:
            t = t - dt
            dt = dt/10
            if dt > 1e-2:
                t = t + dt
                cs = (all_con['glucose [extracellular]'].iloc[-1] 
                - dt * abs(solution.fluxes[glc_exc_id]) 
                * all_con['biomass [cytoplasm]'].iloc[-1])
            else:
                cs = 0
                dt = 0.5
                t = t+dt
                
        # Find active exchange reactions 
        for reaction in model.reactions:
            reaction_id = reaction.id
            if len(model.reactions.get_by_id(reaction_id).metabolites) == 1: 
                flux = solution.fluxes[reaction_id]
                if abs(flux) > 1e-6:
                    df_flux[reaction_id] = [flux]
                
        # Calculate concentrations
        Mx = all_con['biomass [cytoplasm]'].iloc[-1] * V
        for reaction in df_flux.columns:
            if reaction == 't':
                continue
            elif gecko == False:
                if solution.fluxes[reaction] <= 0:
                    continue
                else:
                    for m in model.reactions.get_by_id(reaction).metabolites:
                        metabolite = m.name
                        try:
                            Mm = all_con[metabolite].iloc[-1] * V
                        except (KeyError) as e:
                            Mm = 0
                        df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
            else:
                if 'REV' in reaction or 'pool' in reaction:
                    continue
                else:
                    for m in model.reactions.get_by_id(reaction).metabolites:
                        metabolite = m.name
                        try:
                            Mm = all_con[metabolite].iloc[-1]*V
                        except (KeyError) as e:
                            Mm = 0
                        df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
        # Calculate ethanol concentration
        if gecko == False:
            df_con['ethanol [extracellular]'] = [all_con['ethanol [extracellular]'].iloc[-1]
                                                + dt*solution.fluxes['r_1761']*Mx/V]
        else:
            df_con['ethanol [extracellular]'] = [all_con['ethanol [extracellular]'].iloc[-1]
                                                + dt*solution.fluxes['r_1761']*Mx/V
                                                - dt*solution.fluxes['r_1761_REV']*Mx/V]
        
        # Save results
        for column in all_con.columns:
            try:
                con = df_con[column].iloc[-1]
                if np.isnan(con):
                    df_con[column] = all_con[column].iloc[-1]
            except (KeyError) as e:
                df_con[column] = all_con[column].iloc[-1]
        df_con['glucose [extracellular]'] = [cs]
        df_flux['t'] = [t]
        df_con['t'] = [t]
        all_con = all_con.append(df_con, sort=False)
        all_flux = all_flux.append(df_flux, sort=False)
   
    return(all_flux, all_con)

# --------------------------------------------------------
# FED-BATCH FUNCTIONS 
# --------------------------------------------------------

def fed_batch(
    model, cs_0, cx_0, V_0, feed, t_end, t_lag,  
    dt=0.5, gecko=True,glc_exc_id='r_1714_REV',
    growth_id='r_2111',qs_max=10, kms=0.28
):
    '''
    Modification of batch function to include media feedig.
    Simulate a given GEM in a fed-batch reactor assuming glc limitation, calculates
    metabolite concentrations in the reactor using Euler´s integration method.
    Ethanol uptake is allowed if ethanol is present in the reactor at a max
    uptake rate of 5 mmol/gDW/h
    Inputs: 
        - model: GEM
        - c_s0: initial glucose concentration in the reactor (mmol/l)
        - c_x0: initial biomass concentration in the reactor (gDW/l)
        - V_0: initial volume of the reactor (L)
        - feed: function that determines the feed rate and cs_in as a function of t
        - t_end: duration of the batch (h)
        - t_lag: time of lag phase (h)
        - dt: time interval used for integration (h)
        - gecko: True if ecGEM, False otherwise
        - glc_exchange_id: id of glucose exchange reaction (REV if gecko)
        - growth_id: id of biomass production reaction
        - qs_max: maximum glucose uptake rate in mmol/gDW/h
        - kms: Michaeilis-Menten constant for glucose uptake in mmol/l      
    Outputs:
        - all_flux: dataframe with fluxes through exchange reactions at each t
        - all_con: dataframe with metabolite concentrations in the reactor at each t
        - all_V: list with all the values for volume at different t (L)        
    NOTE: the function changes glc and ethanol upper bounds
    '''
    # Create dataframe to store data
    all_con = pd.DataFrame({'t': [0],
                            'glucose [extracellular]': [cs_0],
                            'biomass [cytoplasm]': [cx_0],
                            'ethanol [extracellular]': [0]
                              })
    all_flux = pd.DataFrame({'t': [0]})
    all_V = [V_0]
    cs = all_con['glucose [extracellular]'].iloc[-1]
    t = t_lag
    while t < t_end:
        t = t + dt
        df_flux = pd.DataFrame()
        df_con = pd.DataFrame()
        
        # Calculate qs and set as lower/upper bound             
        F, cs_in = feed(t)
        qs_mm = qs_max * cs/(kms+cs)
        qs_mb = F * cs_in / (all_con['biomass [cytoplasm]'].iloc[-1] * all_V[-1])
        if F > 0:
            qs = min(qs_mm, qs_mb)
        else:
            qs = qs_mm

        
        if gecko == False:
            model.reactions.get_by_id(glc_exc_id).lower_bound = -qs
        else:
            if qs < 0:
                qs = 0
            model.reactions.get_by_id(glc_exc_id).upper_bound = qs
            
        # ETH uptake
        E = all_con['ethanol [extracellular]'].iloc[-1]
        if gecko == True:
            if E > 0:
                model.reactions.r_1761_REV.upper_bound = 5
            else:
                model.reactions.r_1761_REV.upper_bound = 0
            
        # Perform FBA
        model.objective = growth_id
        model.objective.direction = 'max'
        try:
            solution = model.optimize()
        except (UserWarning) as e:
            print('Solution infeasible, t =', t)
            break

        # Calculate change in volume
        V = all_V[-1] + F*dt
        
        # Check cs
        Ms = (all_con['glucose [extracellular]'].iloc[-1] * all_V[-1] 
              - dt * abs(solution.fluxes[glc_exc_id]) 
              * all_con['biomass [cytoplasm]'].iloc[-1] * all_V[-1]
              + F * cs_in * dt)
        
        cs = Ms/V
        if cs > 0:
            cs = cs
        else:
            t = t - dt
            dt = dt/10
            if dt > 1e-2:
                t = t + dt
                cs = (all_con['glucose [extracellular]'].iloc[-1] 
                - dt * abs(solution.fluxes[glc_exc_id]) 
                * all_con['biomass [cytoplasm]'].iloc[-1])
            else:
                cs = 0
                dt = 0.5
                t = t+dt
                
        # Find active exchange reactions 
        for reaction in model.reactions:
            reaction_id = reaction.id
            if len(model.reactions.get_by_id(reaction_id).metabolites) == 1: #Only exchange reactions
                flux = solution.fluxes[reaction_id]
                if abs(flux) > 1e-6:
                    df_flux[reaction_id] = [flux]
                
        # Calculate concentrations
        Mx = all_con['biomass [cytoplasm]'].iloc[-1] * all_V[-1]
        for reaction in df_flux.columns:
            if reaction == 't':
                continue
            elif gecko == False:
                if solution.fluxes[reaction] <= 0:
                    continue
                else:
                    for m in model.reactions.get_by_id(reaction).metabolites:
                        metabolite = m.name
                        try:
                            Mm = all_con[metabolite].iloc[-1] * all_V[-1]
                        except (KeyError) as e:
                            Mm = 0
                        df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
            else:
                if 'REV' in reaction or 'pool' in reaction:
                    continue
                else:
                    for m in model.reactions.get_by_id(reaction).metabolites:
                        metabolite = m.name
                        try:
                            Mm = all_con[metabolite].iloc[-1]*all_V[-1]
                        except (KeyError) as e:
                            Mm = 0
                        df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
        # Calculate ethanol concentration
        if gecko == False:
            df_con['ethanol [extracellular]'] = [all_con['ethanol [extracellular]'].iloc[-1]
                                                + dt*solution.fluxes['r_1761']*Mx/V]
        else:
            df_con['ethanol [extracellular]'] = [all_con['ethanol [extracellular]'].iloc[-1]
                                                + dt*solution.fluxes['r_1761']*Mx/V
                                                - dt*solution.fluxes['r_1761_REV']*Mx/V]
        
        # Save results
        for column in all_con.columns:
            try:
                con = df_con[column].iloc[-1]
                if np.isnan(con):
                    df_con[column] = all_con[column].iloc[-1]
            except (KeyError) as e:
                df_con[column] = all_con[column].iloc[-1]*all_V[-1]/V
        df_con['glucose [extracellular]'] = [cs]
        df_flux['t'] = [t]
        df_con['t'] = [t]
        all_con = all_con.append(df_con, sort=False)
        all_flux = all_flux.append(df_flux, sort=False)
        all_V.append(V)
        
    return(all_flux, all_con, all_V)

# --------------------------------------------------------
# BATCH WITH MULTIPLE CARBON SOURVES FUNCTIONS 
# --------------------------------------------------------

def batch_multipleC(
    model,
    gecko,
    growth_id,
    dic_sub, 
    cx_0,
    V,
    dt, 
    t_end
):
    '''
    Modification of batch function to include multiple carbon sources.
    Simulate a given GEM in a batch reactorwith multiple C sources, calculates
    metabolite concentrations in the reactor using Euler´s integration method.
    Inputs: 
        - model: GEM
        - gecko: True if ecGEM, False otherwise
        - growth_id: id of biomass production reaction
        - dic_sub: dictionary:
            - key: substrate name as defined in the GEM
            - value: list containing:
                - The exchange reaction id for that substrate
                - Initial concentration of that substrate (mmol/l)
        - c_x0: initial biomass concentration in the reactor (gDW/l)
        - V: volume of the reactor (L)
        - dt: time interval used for Euler's integration (l)
        - t_end: duration of the batch (h)
    Outputs:
        - all_flux: dataframe with fluxes through exchange reactions at each t
        - all_con: dataframe with metabolite concentrations in the reactor at each t  
    '''
    
    # Create dataframe to store data
    all_con = pd.DataFrame({'t': [0],
                            'biomass [cytoplasm]': [cx_0],
                            'ethanol [extracellular]': [0]
                              })
    for substrate in dic_sub:
        all_con[substrate] = dic_sub[substrate][1]
    all_flux = pd.DataFrame({'t': [0]})
    t = 0
   
    while t < t_end:
        t = t + dt
        df_flux = pd.DataFrame()
        df_con = pd.DataFrame()
        
        # Calculate qs and set as lower/upper bound
        if gecko == False:
            for substrate in dic_sub:
                r_id = dic_sub[substrate][0]
                if all_con[substrate].iloc[-1] <= 0:                
                    model.reactions.get_by_id(r_id).lower_bound = 0
                else:
                    model.reactions.get_by_id(r_id).lower_bound = -10
                    if substrate == 'D-glucose [extracellular]':
                        c_glc = all_con[substrate].iloc[-1]
                        qs_max = 20
                        kms = 0.28
                        qs_mm = qs_max*c_glc / (kms+c_glc)
        else:
            for substrate in dic_sub:
                r_id = dic_sub[substrate][0]
                if all_con[substrate].iloc[-1] <= 0:                
                    model.reactions.get_by_id(r_id).upper_bound = 0
                else:
                    model.reactions.get_by_id(r_id).upper_bound = 100
                    if substrate == 'D-glucose [extracellular]':
                        c_glc = all_con[substrate].iloc[-1]
                        qs_max = 20
                        kms = 0.28
                        qs_mm = qs_max*c_glc / (kms+c_glc)

        # Perform FBA
        model.objective = growth_id
        model.objective.direction = 'max'
        try:
            solution = model.optimize()
        except (UserWarning) as e:
            print('Solution infeasible, t =', t)
            break
                
        # Find active exchange reactions 
        for reaction in model.reactions:
            reaction_id = reaction.id
            if len(model.reactions.get_by_id(reaction_id).metabolites) == 1: #Only exchange reactions
                flux = solution.fluxes[reaction_id]
                if abs(flux) > 1e-6:
                    df_flux[reaction_id] = [flux]
                
        # Calculate concentrations
        Mx = all_con['biomass [cytoplasm]'].iloc[-1] * V
        for reaction in df_flux.columns:
            if reaction == 't':
                continue
            elif solution.fluxes[reaction] <= 0:
                continue
            else:
                for m in model.reactions.get_by_id(reaction).metabolites:
                    metabolite = m.name
                    try:
                        Mm = all_con[metabolite].iloc[-1]*V
                    except (KeyError) as e:
                        Mm = 0
                    df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
        
        # Calculate substrate concentrations
        if gecko == False:
            for substrate in dic_sub:
                Mm = all_con[substrate].iloc[-1]*V
                reaction = dic_sub[substrate][0]
                Mm_new = (Mm + dt*solution.fluxes[reaction]*Mx)
                if substrate == 'D-fructose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_1709']*Mx
                elif substrate == 'D-glucose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_1714']*Mx
                elif substrate == 'sucrose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_2058']*Mx
                elif substrate == 'D-mannose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_1715']*Mx

                if Mm_new < 0.01:
                    print(f'Negative {substrate} concentration ({Mm_new}) at t = {t}')
                    Mm_new = 0                
                df_con[substrate] = [Mm_new/V]
                
        else: 
            for substrate in dic_sub:
                Mm = all_con[substrate].iloc[-1]*V
                reaction = dic_sub[substrate][0]
                Mm_new = (Mm - dt*solution.fluxes[reaction]*Mx)
                if substrate == 'D-fructose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_1709']*Mx
                elif substrate == 'D-glucose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_1714']*Mx
                elif substrate == 'sucrose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_2058']*Mx
                elif substrate == 'D-mannose [extracellular]':
                    Mm_new += dt*solution.fluxes['r_1715']*Mx

                if Mm_new < 0:
                    print(f'Negative {substrate} concentration ({Mm_new}) at t = {t}')
                    Mm_new = 0                
                df_con[substrate] = [Mm_new/V]
        
        # Save results
        for column in all_con.columns:
            try:
                con = df_con[column].iloc[-1]
                if np.isnan(con):
                    df_con[column] = all_con[column].iloc[-1]
            except (KeyError) as e:
                df_con[column] = all_con[column].iloc[-1]
        df_flux['t'] = [t]
        df_con['t'] = [t]
        all_con = all_con.append(df_con, sort=False)
        all_flux = all_flux.append(df_flux, sort=False)
        
    return(all_flux, all_con)


def batch_multipleC_reg(
    model,
    growth_id,
    dic_sub, 
    cx_0,
    V,
    dt,
    t_end
):
    '''
    Modification of batch_multipleC function to include competitive inhibition
    among substrates.
    Simulate a given enzyme constrained GEM in a batch reactorwith multiple C 
    sources, calculates metabolite concentrations in the reactor using Euler´s 
    integration method.
    Inputs: 
        - model: GEM
        - growth_id: id of biomass production reaction
        - dic_sub: dictionary:
            - key: substrate name as defined in the GEM
            - value: list containing:
                - The exchange reaction id for that substrate
                - Initial concentration of that substrate (mmol/l)
        - c_x0: initial biomass concentration in the reactor (gDW/l)
        - V: volume of the reactor (L)
        - dt: time interval used for Euler's integration (l)
        - t_end: duration of the batch (h)
    Outputs:
        - all_flux: dataframe with fluxes through exchange reactions at each t
        - all_con: dataframe with metabolite concentrations in the reactor at each t
    '''
    
    # Create dataframe to store data
    all_con = pd.DataFrame({'t': [0],
                            'biomass [cytoplasm]': [cx_0],
                            'ethanol [extracellular]': [0]
                              })
    for substrate in dic_sub:
        all_con[substrate] = dic_sub[substrate][1]
    all_flux = pd.DataFrame({'t': [0]})
    t = 0
    
    # Avoid sucrose export
    model.reactions.r_2058.bounds = 0,0
    
    while t < t_end:
        t = t + dt
        df_flux = pd.DataFrame()
        df_con = pd.DataFrame()
        
        # Calculate qs and set as lower/upper bound considering COMPETITIVE INHIBITION
        # Initially  uncostrain all reactions
        for substrate in dic_sub:
            r_id = dic_sub[substrate][0]
            model.reactions.get_by_id(r_id).bounds = 0, 100
        model.reactions.r_1166.bounds = 0, 100 #glc transport
        model.reactions.r_1134.bounds = 0, 100        
        # When the cell consumes glc initially, it acumulates FRU and vice versa
        if all_con['D-glucose [extracellular]'].iloc[0] > 0:
            suc_lb = 3.5
            if all_con['D-glucose [extracellular]'].iloc[-1] > 1:
                model.reactions.r_1709_REV.upper_bound = 0
            else:
                model.reactions.r_1709_REV.upper_bound = 100
                
        elif all_con['D-fructose [extracellular]'].iloc[0] > 0:
            suc_lb = 3.5
            if all_con['D-fructose [extracellular]'].iloc[-1] > 1:
                model.reactions.r_1714_REV.upper_bound = 0
                model.reactions.r_1166.upper_bound = 0 #glc transport
            else:
                model.reactions.r_1166.upper_bound = 100 #glc transport
                model.reactions.r_1714_REV.upper_bound = 100
        # If no glucose or fructose at t0, suc hydrolysis faster than uptake
        # Force mannose consumption after sucrose depletion
        else: 
            suc_lb = 20
            if  all_con['sucrose [extracellular]'].iloc[-1] > 0:
                model.reactions.r_1166.upper_bound = 0
                model.reactions.r_1134.upper_bound = 0
            else:
                model.reactions.r_1715_REV.lower_bound = 10 
                model.reactions.r_1166.upper_bound = 100
                model.reactions.r_1134.upper_bound = 100
            
        # Glucose and fructose inhibit sucrose hydrolysis at a rate dependent
        # on their presence at t0    
        if (
            all_con['D-glucose [extracellular]'].iloc[-1] < 15/180e-3 and
            all_con['D-fructose [extracellular]'].iloc[-1] < 15/180e-3
        ):
            fruc = all_con['D-fructose [extracellular]'].iloc[-1]*180e-3
            model.reactions.r_2058_REV.bounds = suc_lb, 100
        else:
            fruc = all_con['D-fructose [extracellular]'].iloc[-1]*180e-3
            model.reactions.r_2058_REV.bounds = 0,0
         
        # Most important: No sugar, no uptake
        for substrate in dic_sub:
            r_id = dic_sub[substrate][0]
            if all_con[substrate].iloc[-1] <= 0:                
                model.reactions.get_by_id(r_id).lower_bound = 0
        if all_con['sucrose [extracellular]'].iloc[-1] <= 0:
            model.reactions.r_2058_REV.lower_bound = 0
                            

        # Perform FBA
        model.objective = growth_id
        model.objective.direction = 'max'
        try:
            solution = model.optimize()
        except (UserWarning) as e:
            for substrate in dic_sub:
                r_id = dic_sub[substrate][0]
                model.reactions.get_by_id(r_id).bounds = 0, 100
            model.reactions.r_1166.bounds = 0, 100 #glc transport
            print('Solution infeasible, t =', t)
            break
                
        # Find active exchange reactions 
        for reaction in model.reactions:
            reaction_id = reaction.id
            if len(model.reactions.get_by_id(reaction_id).metabolites) == 1: #Only exchange reactions
                flux = solution.fluxes[reaction_id]
                if abs(flux) > 1e-6:
                    df_flux[reaction_id] = [flux]
                
        # Calculate concentrations
        Mx = all_con['biomass [cytoplasm]'].iloc[-1] * V
        for reaction in df_flux.columns:
            if reaction == 't':
                continue
            elif 'REV' in reaction or 'pool' in reaction:
                continue
            else:
                for m in model.reactions.get_by_id(reaction).metabolites:
                    metabolite = m.name
                    try:
                        Mm = all_con[metabolite].iloc[-1]*V
                    except (KeyError) as e:
                        Mm = 0
                    df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
        
        # Calculate substrate concentrations
        for substrate in dic_sub:
            Mm = all_con[substrate].iloc[-1]*V
            reaction = dic_sub[substrate][0]
            Mm_new = (Mm - dt*solution.fluxes[reaction]*Mx)
            if substrate == 'D-fructose [extracellular]':
                Mm_new += dt*solution.fluxes['r_1709']*Mx               
            elif substrate == 'D-glucose [extracellular]':
                Mm_new += dt*solution.fluxes['r_1714']*Mx
            elif substrate == 'sucrose [extracellular]':
                Mm_new += dt*solution.fluxes['r_2058']*Mx
            if Mm_new < 0:
                print(f'{substrate} concentration ({Mm_new}) at t = {t}, fixed to 0')
                Mm_new = 0                
            df_con[substrate] = [Mm_new/V]
        
        # Save results
        for column in all_con.columns:
            try:
                con = df_con[column].iloc[-1]
                if np.isnan(con):
                    df_con[column] = all_con[column].iloc[-1]
            except (KeyError) as e:
                df_con[column] = all_con[column].iloc[-1]
        df_flux['t'] = [t]
        df_con['t'] = [t]
        all_con = all_con.append(df_con, sort=False)
        all_flux = all_flux.append(df_flux, sort=False)
        
    return(all_flux, all_con)

# ------------------------------------------------------------
# BATCH WITH OXYGEN LIMITATION FUNCTIONS (lactate production)
# ------------------------------------------------------------
def batch_o2lim(model, gecko, cs_0, cx_0, V, dt, glc_exc_id, growth_id, qs_max, kms, t_end):
    '''
    Based on batch function.
    Simulate a given GEM in a batch reactor with a glucose pulse after 80h
    assuming glc limitation, and oxygen limitation after 24h. Calculates
    metabolite concentrations in the reactor using Euler´s integration method.
    Inputs: 
        - model: GEM
        - gecko: True if ecGEM, False otherwise
        - c_s0: initial glucose concentration in the reactor (mmol/l)
        - c_x0: initial biomass concentration in the reactor (gDW/l)
        - V: volume of the reactor (L)
        - dt: time interval used for integration (h)
        - glc_exchange_id: id of glucose exchange reaction (REV if gecko)
        - growth_id: id of biomass production reaction
        - qs_max: maximum glucose uptake rate in mmol/gDW/h
        - kms: Michaeilis-Menten constant for glucose uptake in mmol/l
        - t_end: duration of the batch (h)
    Outputs:
        - all_flux: dataframe with fluxes through exchange reactions at each t
        - all_con: dataframe with metabolite concentrations in the reactor at each t  
    '''
    # Create dataframe to store data
    all_con = pd.DataFrame({'t': [0],
                            'glucose [extracellular]': [cs_0],
                            'biomass [cytoplasm]': [cx_0],
                            'ethanol [extracellular]': [0]
                              })
    all_flux = pd.DataFrame({'t': [0]})
    cs = all_con['glucose [extracellular]'].iloc[-1]
    t = 0
    while t < t_end:
        t = t + dt
        df_flux = pd.DataFrame()
        df_con = pd.DataFrame()
        
        # Glucose pulses
        if t == 80:
            cs = cs + cs_0
            Ms_pulse = cs_0*V
        else:
            Ms_pulse = 0
        
        # Calculate qs and set as lower/upper bound             
        qs = qs_max * cs/(kms+cs)
        if gecko == False:
            model.reactions.get_by_id(glc_exc_id).lower_bound = -qs
        else:
            if qs < 0:
                qs = 0
            model.reactions.get_by_id(glc_exc_id).upper_bound = qs
            
        # O2 LIMITATION
        if t < 24:
            if gecko == False:
                model.reactions.r_1992.lower_bound = -100
            else:
                model.reactions.r_1992_REV.upper_bound = 100
        else:
            qo_max = (0.12/3.05)/(all_con['biomass [cytoplasm]'].iloc[-1] * V)
            if gecko == False:
                model.reactions.r_1992.lower_bound = -qo_max
            else:
                model.reactions.r_1992_REV.upper_bound = qo_max

        # Perform FBA
        model.objective = growth_id
        model.objective.direction = 'max'
        try:
            solution = model.optimize()
        except (UserWarning) as e:
            print('Solution infeasible, t =', t)
            break
        
        # Check cs
        Ms = (all_con['glucose [extracellular]'].iloc[-1] * V 
              - dt * abs(solution.fluxes[glc_exc_id]) 
              * all_con['biomass [cytoplasm]'].iloc[-1] * V
              + Ms_pulse)
        
        cs = Ms/V
        if cs > 0:
            cs = cs
        else:
            t = t - dt
            dt = dt/10
            if dt > 1e-2:
                t = t + dt
                cs = (all_con['glucose [extracellular]'].iloc[-1] 
                - dt * abs(solution.fluxes[glc_exc_id]) 
                * all_con['biomass [cytoplasm]'].iloc[-1])
            else:
                cs = 0
                dt = 0.5
                t = t+dt
                
        # Find active exchange reactions 
        for reaction in model.reactions:
            reaction_id = reaction.id
            if len(model.reactions.get_by_id(reaction_id).metabolites) == 1: #Only exchange reactions
                flux = solution.fluxes[reaction_id]
                if abs(flux) > 1e-6:
                    df_flux[reaction_id] = [flux]
                
        # Calculate concentrations
        Mx = all_con['biomass [cytoplasm]'].iloc[-1] * V
        for reaction in df_flux.columns:
            if reaction == 't':
                continue
            elif gecko == False:
                if solution.fluxes[reaction] <= 0:
                    continue
                else:
                    for m in model.reactions.get_by_id(reaction).metabolites:
                        metabolite = m.name
                        try:
                            Mm = all_con[metabolite].iloc[-1] *V
                        except (KeyError) as e:
                            Mm = 0
                        df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
            else:
                if 'REV' in reaction or 'pool' in reaction:
                    continue
                else:
                    for m in model.reactions.get_by_id(reaction).metabolites:
                        metabolite = m.name
                        try:
                            Mm = all_con[metabolite].iloc[-1]*V
                        except (KeyError) as e:
                            Mm = 0
                        df_con[metabolite] = [(Mm + dt*solution.fluxes[reaction]*Mx)/V]
                        
        # Calculate ethanol concentration
        if gecko == False:
            df_con['ethanol [extracellular]'] = [all_con['ethanol [extracellular]'].iloc[-1]
                                                + dt*solution.fluxes['r_1761']*Mx/V]
        else:
            df_con['ethanol [extracellular]'] = [all_con['ethanol [extracellular]'].iloc[-1]
                                                + dt*solution.fluxes['r_1761']*Mx/V
                                                - dt*solution.fluxes['r_1761_REV']*Mx/V]
        
        # Save results
        for column in all_con.columns:
            try:
                con = df_con[column].iloc[-1]
                if np.isnan(con):
                    df_con[column] = all_con[column].iloc[-1]
            except (KeyError) as e:
                df_con[column] = all_con[column].iloc[-1]
        df_con['glucose [extracellular]'] = [cs]
        df_flux['t'] = [t]
        df_con['t'] = [t]
        all_con = all_con.append(df_con, sort=False)
        all_flux = all_flux.append(df_flux, sort=False)
        
    return(all_flux, all_con)
