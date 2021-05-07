import numpy as np
import matplotlib.pyplot as plt

def scalar_metrics(writer, metrics, total_steps):
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_steps)