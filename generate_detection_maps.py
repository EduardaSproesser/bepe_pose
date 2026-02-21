"""
Script para gerar mapas de detecção para todas as configurações
"""
import pandas as pd
import os
from graphs import DrawGraphs

# Configurações
base_path = r"C:\Users\eduar\OneDrive\Área de Trabalho\bepe\codes\markers\data\d50\results"
marker_types = ["1p", "2e", "3e", "2v", "3v"]
estimation_types = ["single", "multi_iterative", "multi_mean"]
results_folder = "results"  # Pasta base para todos os resultados

print("="*80)
print("GERANDO MAPAS DE DETECÇÃO (TABELA) PARA TODAS AS CONFIGURAÇÕES")
print("="*80)

for marker_type in marker_types:
    for estimation_type in estimation_types:
        print(f"\nProcessando {marker_type} - {estimation_type}...")
        
        try:
            # Construir caminhos dos arquivos CSV
            csv_file1 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_1_with_poses_{estimation_type}.csv")
            csv_file2 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_2_with_poses_{estimation_type}.csv")
            csv_file3 = os.path.join(base_path, f"corners_{marker_type}_{marker_type}_3_with_poses_{estimation_type}.csv")
            
            # Verificar se os arquivos existem
            if not all(os.path.exists(f) for f in [csv_file1, csv_file2, csv_file3]):
                print(f"  ⚠️  Arquivos não encontrados, pulando...")
                continue
            
            # Combinar CSVs
            df1 = pd.read_csv(csv_file1, sep=';')
            df2 = pd.read_csv(csv_file2, sep=';')
            df3 = pd.read_csv(csv_file3, sep=';')
            combined_df = pd.concat([df1, df2, df3], ignore_index=True)
            
            # Salvar CSV combinado temporário
            temp_csv = f"temp_combined_{marker_type}_{estimation_type}.csv"
            combined_df.to_csv(temp_csv, sep=';', index=False)
            
            # Criar instância e gerar visualizações
            graph_drawer = DrawGraphs(temp_csv, estimation_type=estimation_type, 
                                     marker_type=marker_type, save_folder=results_folder)
            
            # Gerar mapa em formato de tabela (distância vs ângulo)
            print(f"  ✓ Gerando mapa de detecção (tabela distância × ângulo)...")
            graph_drawer.plot_detection_heatmap_3d()
            
            # Remover arquivo temporário
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
                
        except Exception as e:
            print(f"  ❌ Erro: {str(e)}")
            continue

print("\n" + "="*80)
print("✅ MAPAS DE DETECÇÃO GERADOS COM SUCESSO!")
print(f"📁 Resultados salvos em: {results_folder}/")
print("="*80)
