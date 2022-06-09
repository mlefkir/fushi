import xspec

multiplicative_models = ["SSS_ice","constant","ismabs"," pwab","wndabs","zphabs",
                       "TBabs","cyclabs","ismdust","redden","xion","zredden",
                       "TBfeo","dust","log10con","smedge","xscat","zsmdust",
                       "TBgas","edge","logconst","spexpcut","zTBabs","zvarabs",
                       "TBgrain","expabs","lyman","spline","zbabs","zvfeabs",
                       "TBpcf","expfac","notch","swind1","zdust","zvphabs",
                       "TBrel","gabs","olivineabs","uvred","zedge","zwabs",
                       "TBvarabs","heilin","pcfabs","varabs","zhighect","zwndabs",
                       "absori","highecut","phabs","vphabs","zigm","zxipab",
                       "acisabs","hrefl","plabs","wabs","zpcfabs","zxipcf","cabs"]

convolution_models = ["cflux","ireflect","kyconv","reflect","thcomp","xilconv",
                     "clumin","kdblur","lsmooth","rfxconv","vashift","zashift",
                     "cpflux","kdblur2","partcov","rgsxsrc","vmshift","zmshift",
                     "gsmooth","kerrconv","rdblur","simpl"]

additive_models = ["agauss","c6vmekl","eqpair","nei","rnei","vraymond",
                    "agnsed","carbatm","eqtherm","nlapec","sedov","vrnei",
                    "agnslim","cemekl","equil","npshock","sirf","vsedov",
                    "apec","cevmkl","expdec","nsa","slimbh","vtapec",
                    "bapec","cflow","ezdiskbb","nsagrav","smaug","vvapec",
                    "bbody","compLS","gadem","nsatmos","snapec","vvgnei",
                    "bbodyrad","compPS","gaussian","nsmax","srcut","vvnei",
                    "bexrav","compST","gnei","nsmaxg","sresc","vvnpshock",
                    "bexriv","compTT","grad","nsx","ssa","vvpshock",
                    "bkn2pow","compbb","grbcomp","nteea","step","vvrnei",
                    "bknpower","compmag","grbjet","nthComp","tapec","vvsedov",
                    "bmc","comptb","grbm","optxagn","vapec","vvtapec",
                    "bremss","compth","hatm","optxagnf","vbremss","vvwdem",
                    "brnei","cph","jet","pegpwrlw","vcph","vwdem",
                    "btapec","cplinear","kerrbb","pexmon","vequil","wdem",
                    "bvapec","cutoffpl","kerrd","pexrav","vgadem","zagauss",
                    "bvrnei","disk","kerrdisk","pexriv","vgnei","zbbody",
                    "bvtapec","diskbb","kyrline","plcabs","vmcflow","zbknpower",
                    "bvvapec","diskir","laor","posm","vmeka","zbremss",
                    "bvvrnei","diskline","laor2","powerlaw","vmekal","zcutoffpl",
                    "bvvtapec","diskm","logpar","pshock","vnei","zgauss",
                    "bwcycl","disko","lorentz","qsosed","vnpshock","zkerrbb",
                    "c6mekl","diskpbb","meka","raymond","voigt","zlogpar",
                    "c6pmekl","diskpn","mekal","redge","vpshock","zpowerlw",
                    "c6pvmkl","eplogpar","mkcflow","refsch"]

def get_components_list(expression,delimiters =("+","*")):
    """
    Returns the list of components of the given expression 
    
    inspired from https://stackoverflow.com/a/13184791  
    """ 
    import re
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern,expression.replace("(","*").replace(")","").replace(" ",""))


def get_param_nb(component):
    """
    Counts the number of parameters added by this new component  
    
    Parameter
    ---------
    component : str
        The name of the component
    
    Returns
    -------
    nb_par : int
        The number of parameters added by this component
       
    """
    xspec.AllModels.clear()
    if component in additive_models  or  'atable' in component :
        return xspec.Model(component).nParameters 
    elif component in multiplicative_models or component in convolution_models or "mtable" in component :
        return xspec.Model(f"{component}*powerlaw").nParameters - 2
        
def get_operator(component):
    """
    Returns the operator associated to the component
    
    Parameter
    ---------
    component : str
        The name of the component
    
    Returns
    -------
    operator : str
        The operator associated to the component, either '+' or '*'
    """
    
    if component in additive_models  or  'atable' in component :
        return "+"
    elif component in multiplicative_models or component in convolution_models or "mtable" in component :
        return "*"
    
def str_in_list(a):
    """
    Find if the list contains a string

    Parameter
    ---------
    a : list
        The list to search in
    Returns
    -------
    bool : True if the list contains the string, False otherwise
    """
    for x in a :
        if type(x)==str:
            return True
    return False

def get_comp_id(nb_orig_par):
    """
    Returns a dictonnary where the entry is a parameter id and the value is the component_id-1
    
    """
    
    par_comp = {}
    par_id = 1
    for k in range(len(nb_orig_par)):
        for w in range(nb_orig_par[k]):
            par_comp[f"{par_id}"] = k
            par_id += 1
    return par_comp


def addComp(add_index,component):
    """
    Add a component at the given index in the model 
    
    Parameters
    ----------
    add_index : int
        The index where to add the component
        1 means to replace the first component, 2 the second, etc.
    component : str
        The name of the component to add
    """
    xspec.Xset.chatter = 0
    expression = xspec.AllModels(1).expression
    nb_par_init = xspec.AllModels(1).nParameters
    
    # save values from model
    old_values = []
    for i in range(1,xspec.AllData.nGroups+1):
        old_values.append([])    
        for par_id in range(1,xspec.AllModels(i).nParameters+1):
            if xspec.AllModels(i)(par_id).link == "" :
                old_values[i-1].append(xspec.AllModels(i)(par_id).values)
            else : 
                old_values[i-1].append(xspec.AllModels(i)(par_id).link)
    
    components_list =  get_components_list(expression) # list of components
    assert add_index<=len(components_list), "Add index is greater than the number components!"
    nb_added_par = get_param_nb(component) # counts the number of parameters added
    nb_orig_par = [get_param_nb(original_component) for original_component in components_list ] # stores the number of parameters for each component
    assert sum(nb_orig_par)==nb_par_init,"The number of parameters calculated differs from original number of parameters ! "
    par_comp_dict = get_comp_id(nb_orig_par)
    
    # generate the new expression
    new_expression = expression.replace(components_list[add_index-1],f"{component} {get_operator(component)} {components_list[add_index-1]}")
    new_nb_par = nb_added_par+sum(nb_orig_par)
    
    # load the new model
    xspec.AllModels.clear()
    model = xspec.Model(new_expression)
    nb_orig_par.insert(add_index-1,f"{nb_added_par}")

    
    for i in range(1,xspec.AllData.nGroups+1):
        mod_id = 0
        par_id = 1
        orig_id = 0
        completed = []

        while par_id <= new_nb_par :
            if type(nb_orig_par[mod_id]) == str :
                par_id += int(nb_orig_par[mod_id])
                completed.append(nb_orig_par[mod_id])
                mod_id += 1
            else :
                if len(completed) < mod_id + 1 :
                    completed.append(1)
                elif completed[mod_id] < nb_orig_par[mod_id] :
                    completed[mod_id] += 1
                if completed[mod_id] == nb_orig_par[mod_id] :
                    mod_id += 1

                if type(old_values[i-1][orig_id]) == str :
                    if int(f"{old_values[i-1][orig_id]}"[3:]) > nb_par_init: # in case there is a free parameter after the first group
                        if add_index-1 >= mod_id :
                            xspec.AllModels(i)(par_id).link = f"{int(f'{old_values[i-1][orig_id]}'[3:])+(i-2)*nb_added_par}"
                        else :
                            xspec.AllModels(i)(par_id).link = f"{int(f'{old_values[i-1][orig_id]}'[3:])+(i-1)*nb_added_par}"
                    else :
                        if add_index <= par_comp_dict[old_values[i-1][orig_id][3:]]+1 : #str_in_list(nb_orig_par[:mod_id]) and i>1 :
                            xspec.AllModels(i)(par_id).link =  f"{int(f'{old_values[i-1][orig_id]}'[3:])+nb_added_par}"
                        #elif add_index < par_comp_dict[old_values[i-1][orig_id][3:]]: 
                        #    xspec.AllModels(i)(par_id).link =  f"{int(f'{old_values[i-1][orig_id]}'[3:])+nb_added_par}"
                        else : 
                            xspec.AllModels(i)(par_id).link =  f"{int(f'{old_values[i-1][orig_id]}'[3:])}"

                else :
                    xspec.AllModels(i)(par_id).values = old_values[i-1][orig_id]
                    xspec.AllModels(i)(par_id).frozen = True 
                orig_id += 1
                par_id += 1

def old_addComp(component):
    """
    Add a component at the end of the model in PyXspec
    
    Parameters
    ----------
    component : str
        Name of the component
    """
    
    expression = xspec.AllModels(1).expression
    nb_par_init = xspec.AllModels(1).nParameters

    # save values from model
    old_values = []
    for i in range(1,xspec.AllData.nGroups+1):
        old_values.append([])    
        for par_id in range(1,xspec.AllModels(i).nParameters+1):
            if xspec.AllModels(i)(par_id).link == "" :
                old_values[i-1].append(xspec.AllModels(i)(par_id).values)
            else : 
                old_values[i-1].append(xspec.AllModels(i)(par_id).link)

    cstat_bf = xspec.Fit.statistic
    
    xspec.AllModels.clear()
    nb_added_par = xspec.Model(component).nParameters # counts the number of parameters added
    
    xspec.AllModels.clear()
    new_expr = expression.replace(')',f' + {component} )')
    model = xspec.Model(new_expr)

    # restore old values 
    for i in range(1,xspec.AllData.nGroups+1):
        for par_id in range(1,len(old_values[i-1])+1):
            if type(old_values[i-1][par_id-1]) == str :
                if int(f"{old_values[i-1][par_id-1]}"[3:]) > nb_par_init: # in case there is a free parameter after the first group
                    xspec.AllModels(i)(par_id).link = f"{int(f'{old_values[i-1][par_id-1]}'[3:])+(i-2)*nb_added_par}"
                else :
                    xspec.AllModels(i)(par_id).link = old_values[i-1][par_id-1][3:]
            else :
                xspec.AllModels(i)(par_id).values = old_values[i-1][par_id-1]
                xspec.AllModels(i)(par_id).frozen = True

    return cstat_bf
