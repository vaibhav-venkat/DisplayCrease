import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import json

CSV_START_TOKEN = "__CREASE_CSV_BEGIN__"
CSV_END_TOKEN = "__CREASE_CSV_END__"

print("Loading GA modules...", flush=True)
import crease2d_ga_functions as crease2d_ga
print("[OK] All modules loaded", flush=True)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Initializing Genetic Algorithm...", flush=True)

if len(sys.argv) < 2:
    print("Error: Input sample id or filename is required.", flush=True)
    print("Usage: python GA_py_script_modular.py <sample_id_or_filename> [model_type]", flush=True)
    print("  model_type: 'hollowTubes' or 'Ellipsoids' (default: hollowTubes)", flush=True)
    quit()

# Get model type (default to hollowTubes for backward compatibility)
model_type = sys.argv[2] if len(sys.argv) > 2 else 'hollowTubes'
print(f"Model Type: {model_type}", flush=True)

print(f"Loading {model_type} configuration...", flush=True)
try:
    model_config = crease2d_ga.get_model_config(model_type)
    print(f"[OK] Model configuration loaded", flush=True)
    print(f"  - Number of genes: {model_config.numgenes}", flush=True)
    print(f"  - Default Q-theta grid: {model_config.nq} × {model_config.ntheta} (will adjust to input)", flush=True)
except ValueError as e:
    print(f"Error: {e}", flush=True)
    quit()

print("Loading XGBoost model...", flush=True)
try:
    loaded_model = crease2d_ga.load_model(model_type, models_dir='../models')
    
    # Try to use CUDA if available
    try:
        config_dict = json.loads(loaded_model.save_config())
        config_dict['learner']['generic_param']['device'] = 'cuda'
        loaded_model.load_config(json.dumps(config_dict))
        print(f"[OK] XGBoost model loaded (Device: cuda)", flush=True)
    except:
        print(f"[OK] XGBoost model loaded (Device: cpu)", flush=True)
except Exception as e:
    print(f"Error loading model: {e}", flush=True)
    quit()

print("Loading input data...", flush=True)
input_arg = sys.argv[1]

# Determine data directory based on model type
if model_type.lower() in ['ellipsoid', 'ellipsoids']:
    data_subdir = 'Ellipsoids'
else:
    data_subdir = 'hollowTubes'

try:
    # Try to parse as sample ID (integer)
    inputsampleid = int(input_arg)
    if model_type.lower() in ['ellipsoid', 'ellipsoids']:
        infilename = f"Ellipsoids_test_sample_{inputsampleid}.txt"
    else:
        infilename = f"sample_{inputsampleid}_orig_datayz.txt"
    sample_name = f"sample_{inputsampleid}"
except ValueError:
    # It's a filename
    infilename = input_arg
    sample_name = os.path.splitext(os.path.basename(infilename))[0]
    sample_name = sample_name.replace('_orig_datayz', '').replace('_datayz', '').replace('Ellipsoids_test_', '')

input_Iq = crease2d_ga.read_Iq(infilename, f"../data/{data_subdir}")
print(f"[OK] Loaded input data from: {infilename}", flush=True)
print(f"  - Input data shape: {input_Iq.shape[0]} (q) × {input_Iq.shape[1]} (theta)", flush=True)

print(f"Updating model configuration to match input data shape...", flush=True)
model_config.update_grid_from_input(input_Iq)
print(f"[OK] Model configuration updated", flush=True)
print(f"  - Q points: {model_config.nq} (range: {model_config.q_min if hasattr(model_config, 'q_min') else f'exp({model_config.q_min_exp})'} to {model_config.q_max if hasattr(model_config, 'q_max') else f'exp({model_config.q_max_exp})'}))", flush=True)
print(f"  - Theta points: {model_config.ntheta} (range: {model_config.theta_min} to {model_config.theta_max})", flush=True)

popsize = 300
numgenes = model_config.numgenes
numGAdatasets = 1

numgens = 100
numparents = 100  # keep 1/3rd of the population for mating
transitionpoint = 250  # Generation after which mutation rate is reduced
numelites_init = 50
numelites_final = 10
mutationrateconst_init = 0.5
mutationrateconst_final = 0.05

# Print GA Configuration
print("\n" + "="*60, flush=True)
print("GENETIC ALGORITHM CONFIGURATION", flush=True)
print("="*60, flush=True)
print(f"Model Type: {model_type}", flush=True)
print(f"Sample: {sample_name}", flush=True)
print(f"Input File: {infilename}", flush=True)
print(f"\nPopulation Settings:", flush=True)
print(f"  - Population Size: {popsize} individuals", flush=True)
print(f"  - Number of Genes: {numgenes}", flush=True)
print(f"  - Number of GA Runs: {numGAdatasets}", flush=True)
print(f"\nEvolution Settings:", flush=True)
print(f"  - Total Generations: {numgens}", flush=True)
print(f"  - Parents per Generation: {numparents}", flush=True)
print(f"  - Transition Point: Generation {transitionpoint}", flush=True)
print(f"\nElitism Settings:", flush=True)
print(f"  - Initial Elites: {numelites_init}", flush=True)
print(f"  - Final Elites: {numelites_final}", flush=True)
print(f"\nMutation Settings:", flush=True)
print(f"  - Initial Mutation Rate: {mutationrateconst_init}", flush=True)
print(f"  - Final Mutation Rate: {mutationrateconst_final}", flush=True)
print(f"\nStructural Features:", flush=True)
for i, title in enumerate(model_config.get_feature_titles(), 1):
    print(f"  {i}. {title}", flush=True)
print("="*60 + "\n", flush=True)

for gaind in range(numGAdatasets):
    GAindex = gaind
    print(f"\n=== Starting GA Run {GAindex} for {model_type}_{sample_name} ===", flush=True)
    
    # Generate random initial population
    print(f"Generating random initial population of {popsize} individuals...", flush=True)
    gatable = np.random.rand(popsize, numgenes)
    # Add fitness column (initialized to zeros)
    gatable = np.hstack((gatable, np.zeros((popsize, 1))))
    print(f"[OK] Generated {popsize} random gene combinations", flush=True)
    
    # Generate profiles for initial population
    print(f"Computing scattering profiles using XGBoost model...", flush=True)
    struc_featurestable, currentprofiles = crease2d_ga.generateallprofiles(
        gatable, loaded_model, model_config
    )
    print(f"[OK] Generated {popsize} scattering profiles", flush=True)
    
    # Update gatable and save output for generation 0
    print(f"Calculating fitness scores...", flush=True)

    gatable = crease2d_ga.updateallfitnesses(gatable, currentprofiles, input_Iq)
    tableindices = np.flipud(gatable[:,numgenes].argsort())  # sort by descending fitness
    gatable = gatable[tableindices]
    currentprofiles = currentprofiles[:,:,tableindices]
    print(f"[OK] Population sorted by fitness", flush=True)

    # Initialize tracking variables
    meanfitness = np.mean(gatable[:,-1])
    stddevfitness = np.std(gatable[:,-1])
    bestfitness = gatable[0,-1]
    worstfitness = gatable[-1,-1]
    diversitymetric = np.mean(np.sum((gatable[:,:-1]-np.mean(gatable[:,:-1],axis=0))**2, axis=1))/np.sqrt(numgenes)
    print(f'Generation 0: Best={bestfitness:.6f}, Average={meanfitness:.6f}', flush=True)
    print(f"Starting evolution for {numgens} generations...", flush=True)
    print("", flush=True)

    fitness_scores = np.array([[0, meanfitness, stddevfitness, bestfitness, worstfitness]])
    evolvedstrucfeatures = np.reshape(
        np.vstack((np.ones([1, numgenes]), struc_featurestable)),
        (popsize+1, 1, numgenes)
    )

    #==========================================================================
    # Perform GA Iterations
    #==========================================================================
    for currentgen in range(1, numgens+1):
        parents = gatable[0:numparents,:]
        children = crease2d_ga.generate_children(parents, popsize, numgenes)
        
        # Adjust parameters based on generation
        if currentgen <= transitionpoint:
            numelites = numelites_init
            mutationrate = mutationrateconst_init * (1 - diversitymetric)**2
        else:
            numelites = numelites_final
            mutationrate = mutationrateconst_final * (1 - diversitymetric)**2
        
        gatable = np.vstack((parents, children))
        gatable = crease2d_ga.applymutations(gatable, numelites, mutationrate)
        
        # Generate profiles and update fitness
        struc_featurestable, currentprofiles = crease2d_ga.generateallprofiles(
            gatable, loaded_model, model_config
        )
        gatable = crease2d_ga.updateallfitnesses(gatable, currentprofiles, input_Iq)
        
        # Sort by fitness
        tableindices = np.flipud(gatable[:,numgenes].argsort())
        gatable = gatable[tableindices]
        
        # Calculate metrics
        meanfitness = np.mean(gatable[:,-1])
        stddevfitness = np.std(gatable[:,-1])
        bestfitness = gatable[0,-1]
        worstfitness = gatable[-1,-1]
        diversitymetric = np.mean(np.sum((gatable[:,:-1]-np.mean(gatable[:,:-1],axis=0))**2, axis=1))/np.sqrt(numgenes)
        fitness_scores = np.append(
            fitness_scores,
            [[currentgen, meanfitness, stddevfitness, bestfitness, worstfitness]],
            axis=0
        )
        
        # Progress reporting
        if currentgen % 10 == 0:
            improvement = bestfitness - fitness_scores[max(0,currentgen-10),3] if currentgen >= 10 else 0
            print(f'Gen {currentgen}/{numgens}: Best={bestfitness:.6f}, Avg={meanfitness:.6f}, MutRate={mutationrate:.4f}, Improvement={improvement:+.6f}', flush=True)
            evolvedstrucfeatures = np.hstack((
                evolvedstrucfeatures,
                np.reshape(
                    np.vstack((np.ones([1,numgenes])*currentgen, struc_featurestable)),
                    (popsize+1, 1, numgenes)
                )
            ))
        
        # Save checkpoints
        if currentgen % 50 == 0:
            print(f"  >> Checkpoint at generation {currentgen}", flush=True)

    evolvedstrucfeatures = np.swapaxes(evolvedstrucfeatures, 0, 1)
    
    # Generate CSV output for web visualization
    print(f"\nGenerating CSV output for web visualization...", flush=True)

    final_gen_genes = gatable[:, :numgenes]
    final_gen_fitness = gatable[:, -1]
    structural_features = model_config.genes_to_struc_features(final_gen_genes)

    records = []
    is_ellipsoid = model_type.lower() in ['ellipsoid', 'ellipsoids']

    for idx, (struct_vals, fitness_val) in enumerate(zip(structural_features, final_gen_fitness)):
        record = {
            "run_id": idx + 1,
            "fitness": float(fitness_val),
        }

        if is_ellipsoid:
            mean_r, std_r, mean_g, std_g, kappa_val, vol_frac = struct_vals
            kappa_safe = float(max(kappa_val, 1e-12))
            major_axis = float(mean_r * max(mean_g, 1e-12))
            minor_axis = float(mean_r / max(mean_g, 1e-12))

            record.update(
                {
                    "MeanR": float(mean_r),
                    "StdR": float(std_r),
                    "MeanG": float(mean_g),
                    "StdG": float(std_g),
                    "Kappa": float(kappa_val),
                    "Volume_Fraction": float(vol_frac),
                    "kappa_log": float(np.log10(kappa_safe)),
                    # Legacy/compatibility columns
                    "D_A": float(2.0 * mean_r),
                    "e_mu": float(np.nan),
                    "e_sigma": float(np.nan),
                    "omega_deg": float(np.nan),
                    "alpha_deg": float(np.nan),
                    "dh": float(np.nan),
                    "lh": float(np.nan),
                    "nx": float(np.nan),
                    "a": major_axis,
                    "b": minor_axis,
                }
            )
        else:
            dia, ecc_m, ecc_sd, orient, kappa_exp, cone_angle, herd_dia, herd_len, herd_extra = struct_vals

            record.update(
                {
                    "D_A": float(dia),
                    "e_mu": float(ecc_m),
                    "e_sigma": float(ecc_sd),
                    "omega_deg": float(orient),
                    "kappa_log": float(kappa_exp),
                    "alpha_deg": float(cone_angle),
                    "dh": float(herd_dia),
                    "lh": float(herd_len),
                    "nx": int(round(herd_extra)),
                    "a": float(np.nan),
                    "b": float(np.nan),
                    # Placeholders for ellipsoid-specific columns
                    "MeanR": float(np.nan),
                    "StdR": float(np.nan),
                    "MeanG": float(np.nan),
                    "StdG": float(np.nan),
                    "Kappa": float(np.nan),
                    "Volume_Fraction": float(np.nan),
                }
            )

        records.append(record)

    columns = [
        "run_id",
        "D_A",
        "e_mu",
        "e_sigma",
        "omega_deg",
        "kappa_log",
        "alpha_deg",
        "dh",
        "lh",
        "nx",
        "a",
        "b",
        "MeanR",
        "StdR",
        "MeanG",
        "StdG",
        "Kappa",
        "Volume_Fraction",
        "fitness",
    ]

    results_df = pd.DataFrame(records, columns=columns)
    csv_payload = results_df.to_csv(index=False)
    print(CSV_START_TOKEN, flush=True)
    sys.stdout.write(csv_payload)
    if not csv_payload.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    print(CSV_END_TOKEN, flush=True)
    print(f"[OK] CSV results ready for download", flush=True)
    
    print(f"\n[COMPLETED] GA Run {GAindex} finished!", flush=True)
    print(f"  Final Best Fitness: {bestfitness:.6f}", flush=True)
    print(f"  Total Improvement: {bestfitness - fitness_scores[0,3]:+.6f}", flush=True)
    print("  CSV streamed to caller", flush=True)

print("\n" + "="*60, flush=True)
print("GENETIC ALGORITHM COMPLETED SUCCESSFULLY", flush=True)
print("="*60, flush=True)
