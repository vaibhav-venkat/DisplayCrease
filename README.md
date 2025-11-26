# Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050.

## Tabs & Actions

- **Data Preview**: upload scattering matrix via the sidebar, adjust φ/q limits, toggle log transform, click *Recompute Plots*.
- **Model Workspace**: review model parameters and optimization method details after choosing a model + method in the sidebar.
- **Scattering Fit**: compare fitted vs experimental data once a GA run finishes.
- **Parameter Distributions**: review violins from the latest GA run or CREASE CSV (hollow tubes shown in a 3×3 grid, ellipsoids in a 2×3 grid).
- **Processing Logs**: see live GA console output; updates only when this tab is selected.
- **Results**: monitor GA status, read completion summaries, and click *Download GA CSV* for the results.
- **Summary**: stats for the uploaded matrix.
- **Real-Space Reconstruction**: Not done yet. 

## Additional Inputs

- **Upload CREASE CSV** (sidebar) to populate distributions without running GA.
- **Run Optimization** (sidebar button) to start GA with the current model selection. GA polling stops automatically on completion or failure.
