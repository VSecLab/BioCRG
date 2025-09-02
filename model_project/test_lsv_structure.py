#!/usr/bin/env python3
"""
Test script per dimostrare la nuova struttura di dizionari LSV nella funzione segmentation.
"""

import sys
import os
sys.path.append('/Users/grims/Documents/Research/Tesi/ML_tesi/model_project/src')

from segmentation import segmentation_on_activity
from data_processing.src import config

def test_lsv_structure():
    """
    Test per verificare la struttura dei dizionari LSV.
    """
    print("=== Test della struttura dei dizionari LSV ===\n")
    
    # Usa un file di test
    test_file = "/Users/grims/Documents/Research/Tesi/ML_tesi/data_logs/raw/grims/grims_log_20250719_1148_1YAYXWAD50.csv"
    
    if not os.path.exists(test_file):
        print(f"File di test non trovato: {test_file}")
        return
    
    try:
        # Test della funzione segmentation_on_activity con i nuovi parametri di ritorno
        features = config.FEATURES
        logNumber_numberSegments, rsv_on_activity, adjectives_on_activity, lsv_on_activity = segmentation_on_activity(
            file_path=test_file,
            features=features[1:],
            activity="sphereActivity",
            threshold=0.65,
            scaler="standard"
        )
        
        print(f"Numero di log processati: {len(lsv_on_activity)}")
        print(f"Log numbers trovati: {list(lsv_on_activity.keys())}")
        
        # Esplora la struttura del dizionario LSV
        for log_number, lsv_dict in lsv_on_activity.items():
            print(f"\nLog {log_number}:")
            print(f"  Numero di segmenti: {len(lsv_dict)}")
            print(f"  Segmenti: {list(lsv_dict.keys())}")
            
            # Mostra la forma di alcuni vettori LSV
            for segment_num, lsv_vector in list(lsv_dict.items())[:3]:  # Solo primi 3 per brevità
                print(f"    Segmento {segment_num}: LSV shape = {lsv_vector.shape}")
                print(f"    Primi 5 valori: {lsv_vector[:5]}")
        
        print(f"\nStruttura completa:")
        print(f"- logNumber_numberSegments: {logNumber_numberSegments}")
        print(f"- RSV shape: {rsv_on_activity.shape}")
        print(f"- Adjectives shape: {adjectives_on_activity.shape}")
        print(f"- LSV dictionary keys: {list(lsv_on_activity.keys())}")
        
    except Exception as e:
        print(f"Errore durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lsv_structure()
