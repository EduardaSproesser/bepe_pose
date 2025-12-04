import pandas as pd
from find_transformation import FindTransformation

rows = [
    # marker_type A - normal
    {"marker_type":"A","rvec":"0.01,0.02,0.03","tvec":"10,5,3","rvec_est":"0.012,0.018,0.035","tvec_est":"10.3,5.1,3.0"},
    {"marker_type":"A","rvec":"0.02,0.01,0.02","tvec":"12,6,2","rvec_est":"0.018,0.012,0.022","tvec_est":"12.1,6.2,2.1"},
    {"marker_type":"A","rvec":"0.015,0.02,0.01","tvec":"9,4,2","rvec_est":"0.016,0.019,0.009","tvec_est":"8.8,3.9,2.1"},
    # marker_type A - outlier
    {"marker_type":"A","rvec":"1.5,0.0,0.0","tvec":"1000,2000,3000","rvec_est":"1.4,0.1,0.0","tvec_est":"900,1900,2900"},
    # marker_type B - two normal
    {"marker_type":"B","rvec":"0.01,0.01,0.01","tvec":"5,2,1","rvec_est":"0.009,0.011,0.012","tvec_est":"5.1,2.1,1.0"},
    {"marker_type":"B","rvec":"0.012,0.013,0.01","tvec":"5.2,2.1,1.1","rvec_est":"0.013,0.012,0.011","tvec_est":"5.0,2.0,1.05"}
]

df = pd.DataFrame(rows)
csv_path = 'tmp_transform_test.csv'
df.to_csv(csv_path, sep=';', index=False)
print('CSV written to', csv_path)

f = FindTransformation(csv_path)
print('rows loaded:', len(f.data))
mt = f.compute_mean_transformations()
print('mean transforms for markers:', list(mt.keys()))
for k,v in mt.items():
    print(k, '\n', v)
