"""
src/monitoring/simulate_drift.py
--------------------------------
Simulates data drift by inserting synthetic predictions with shifted distributions
into the predictions_log table, allowing the drift detector to flag them.
"""

import datetime
import random
import sys

from src.serving.db import insert_prediction


def simulate_drift(num_records: int = 200, drift_type: str = "amount_shift"):
    """
    Insert records with shifted features to simulate drift.
    `amount_shift`: massively increases the transaction amount.
    """
    print(f"Injecting {num_records} drifted records ({drift_type})...")
    
    now = datetime.datetime.now(datetime.timezone.utc)
    
    for i in range(num_records):
        ts = (now - datetime.timedelta(minutes=num_records - i)).isoformat()
        
        # Simulate normal or shifted data
        if drift_type == "amount_shift":
            # Normal amount is usually < 100. Drifted is around 5000.
            amount = random.uniform(4000.0, 6000.0)
            time_feature = random.uniform(0.0, 170000.0)
        else:
            amount = random.uniform(1.0, 100.0)
            time_feature = random.uniform(0.0, 170000.0)
            
        fraud_prob = random.uniform(0.0, 0.3)
        is_fraud = False
        
        insert_prediction(
            ts=ts,
            fraud_prob=fraud_prob,
            is_fraud=is_fraud,
            model_version="1",
            amount=amount,
            time_feature=time_feature
        )
        
    print("Done injecting synthetic drift.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    else:
        num = 200
        
    simulate_drift(num_records=num)
