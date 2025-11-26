# Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050.

## Tabs 

- **Data Preview**: upload scattering matrix in sidebar, adjust limits, click *Recompute Plots*. Same as your inputs.  
- **Model Workspace**: review model parameters and method details after choosing a model + method in the sidebar. Only support the GA + models provided. 
- **Scattering Fit**: compare fitted vs exp data once a GA run finishes.
- **Parameter Distributions**: review violins from the latest GA run or CREASE CSV 
- **Processing Logs**: see live GA console output; updates only when this tab is selected.
- **Results**: monitor GA status, read completion summaries, and click *Download GA CSV* for the results.
- **Summary**: stats for the uploaded matrix.
- **Real-Space Reconstruction**: Not done yet. 

## Other Inputs

- **Upload CREASE CSV** (sidebar) to populate plots without running GA.
- **Run Optimization** (sidebar button) to start GA with the current model selection. GA runs stops on completion or failure.
