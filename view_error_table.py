import pandas as pd
import numpy as np

# Carregar a tabela
df = pd.read_csv('complete_error_table.csv')

print("="*100)
print("TABELA COMPLETA DE ERROS E DETECÇÕES POR INTERVALO")
print("="*100)
print()

# Mostrar resumo por configuração
print("RESUMO POR CONFIGURAÇÃO (Marker Type + Estimation Type):")
print("-"*100)

for (marker, estimation), group in df.groupby(['Marker_Type', 'Estimation_Type']):
    total_samples = group['Sample_Count'].sum()
    detection_rate_weighted = (
        np.average(group['Detection_Rate'].fillna(0), weights=group['Sample_Count'])
        if total_samples > 0 else np.nan
    )
    print(f"\n{marker} - {estimation}")
    print(f"  Total de bins: {len(group)}")
    print(f"  Erro de translação médio: {group['Mean_Translation_Error'].mean():.3f} ± {group['Mean_Translation_Error'].std():.3f}")
    print(f"  Erro de rotação médio: {group['Mean_Rotation_Error_deg'].mean():.3f}° ± {group['Mean_Rotation_Error_deg'].std():.3f}°")
    print(f"  Taxa de detecção (ponderada): {detection_rate_weighted:.3f}")
    print(f"  Amostras totais: {total_samples:.0f}")

print("\n" + "="*100)
print("ANÁLISE POR TIPO DE BIN:")
print("="*100)

# Por distância
print("\nBINS DE DISTÂNCIA:")
print("-"*100)
dist_bins = df[df['Bin_Type'] == 'Distance']
for range_val in dist_bins['Bin_Range'].unique():
    bin_data = dist_bins[dist_bins['Bin_Range'] == range_val]
    total_samples = bin_data['Sample_Count'].sum()
    if total_samples > 0:
        detection_rate_weighted = np.average(
            bin_data['Detection_Rate'].fillna(0),
            weights=bin_data['Sample_Count']
        )
        print(f"\nIntervalo {range_val} cm:")
        print(f"  Erro translação: {bin_data['Mean_Translation_Error'].mean():.3f} (n={total_samples:.0f})")
        print(f"  Erro rotação: {bin_data['Mean_Rotation_Error_deg'].mean():.3f}°")
        print(f"  Taxa detecção (ponderada): {detection_rate_weighted:.3f}")

# Por ângulo
print("\n\nBINS DE ÂNGULO:")
print("-"*100)
angle_bins = df[df['Bin_Type'] == 'Angle']
for range_val in angle_bins['Bin_Range'].unique():
    bin_data = angle_bins[angle_bins['Bin_Range'] == range_val]
    total_samples = bin_data['Sample_Count'].sum()
    if total_samples > 0:
        detection_rate_weighted = np.average(
            bin_data['Detection_Rate'].fillna(0),
            weights=bin_data['Sample_Count']
        )
        print(f"\nIntervalo {range_val}°:")
        print(f"  Erro translação: {bin_data['Mean_Translation_Error'].mean():.3f} (n={total_samples:.0f})")
        print(f"  Erro rotação: {bin_data['Mean_Rotation_Error_deg'].mean():.3f}°")
        print(f"  Taxa detecção (ponderada): {detection_rate_weighted:.3f}")

print("\n" + "="*100)
print("MELHORES E PIORES CONFIGURAÇÕES:")
print("="*100)

# Agrupar por configuração
config_summary = df.groupby(['Marker_Type', 'Estimation_Type']).agg({
    'Mean_Translation_Error': 'mean',
    'Mean_Rotation_Error_deg': 'mean',
    'Detection_Rate': 'mean',
    'Sample_Count': 'sum'
}).reset_index()
weighted_det = df.groupby(['Marker_Type', 'Estimation_Type']).apply(
    lambda g: np.average(g['Detection_Rate'].fillna(0), weights=g['Sample_Count']) if g['Sample_Count'].sum() > 0 else np.nan
).reset_index(name='Detection_Rate_Weighted')
config_summary = config_summary.merge(weighted_det, on=['Marker_Type', 'Estimation_Type'], how='left')

print("\nMenor erro de translação:")
best_trans = config_summary.nsmallest(3, 'Mean_Translation_Error')
for idx, row in best_trans.iterrows():
    print(f"  {row['Marker_Type']} - {row['Estimation_Type']}: {row['Mean_Translation_Error']:.3f}")

print("\nMenor erro de rotação:")
best_rot = config_summary.nsmallest(3, 'Mean_Rotation_Error_deg')
for idx, row in best_rot.iterrows():
    print(f"  {row['Marker_Type']} - {row['Estimation_Type']}: {row['Mean_Rotation_Error_deg']:.3f}°")

print("\nMaior taxa de detecção (ponderada):")
best_det = config_summary.nlargest(3, 'Detection_Rate_Weighted')
for idx, row in best_det.iterrows():
    print(f"  {row['Marker_Type']} - {row['Estimation_Type']}: {row['Detection_Rate_Weighted']:.3f}")

print("\n" + "="*100)
