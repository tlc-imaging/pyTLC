import pandas as pd
def run_database_search(tlc_dataset, lipid_database):
    ppm = 15.
    max_peaks = 3
    adducts = ["+H", "+Na", "+K"]
    dbase = pd.read_csv(lipid_database)
    print dbase.head()
    feature_list=[]
    for row in dbase.iterrows():
        for adduct in adducts:
            mz = row[1][adduct]
            lipid_name = row[1]['name']
            print lipid_name
            lipid_species = ""
            if '(' in lipid_name:
                print np.where(lipid_name=='(')
                lipid_species = lipid_name[0:np.where(lipid_name=='(')[0][0]]
            print lipid_species
            tol = mz*ppm*1e-6
            xic_features = tlc_dataset.get_xic(mz,tol,min_int=100).get_xic(source='centroids')
            n_peaks = len(xic_features[0])
            if all((n_peaks < max_peaks,)):
                for x, i in zip(xic_features[0], xic_features[1]):
                    feature_list.append((mz, x, i))
    return feature_list