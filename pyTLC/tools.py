import pandas as pd
import numpy as np

def run_database_search(tlc_dataset, lipid_database):
    ppm = 6.
    max_peaks = 3
    adducts = ["+H", "+Na", "+K"]
    dbase = pd.read_csv(lipid_database)
    #print "database head", dbase.head()
    feature_list={}
    for row in dbase.iterrows():
        for adduct in adducts:
            mz = row[1][adduct]
            lipid_name = row[1]['name']
            print "lipid name",lipid_name
            lipid_species = ""
            #if '(' in lipid_name:
            #    print "np.where", np.where(lipid_name=='(')
            #    lipid_species = lipid_name[0:np.where(lipid_name=='(')[0][0]]
            #print lipid_species
            tol = ppm
            xic_features = tlc_dataset.get_xic(mz,tol,min_int=1).get_xic(source='features')
            print "lib_mz:", mz, "found_mz",xic_features[0], "found_int:", xic_features[1]
            n_peaks = len(xic_features[0])
            if all((n_peaks < max_peaks,)):
                for x, i in zip(xic_features[0], xic_features[1]):
                    feature_list[mz] = (x,i)
            
    return feature_list
