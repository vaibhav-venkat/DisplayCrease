#Functions for CREASE-2D GA
import os
import numpy as np
import xgboost as xgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

#=============================================================================
# Base Model Configuration Class
#=============================================================================
class ModelConfig:
    """Base class for model-specific configurations"""
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.setup_qtheta_grid()
    
    def setup_qtheta_grid(self):
        """Setup q-theta grid - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement setup_qtheta_grid")
    
    def update_grid_from_input(self, input_data):
        """Update grid dimensions based on actual input data shape"""
        if input_data is not None:
            self.nq = input_data.shape[0]
            self.ntheta = input_data.shape[1]
            self.nqtheta = self.nq * self.ntheta
            # Regenerate grid with new dimensions
            self._regenerate_grid()
    
    def _regenerate_grid(self):
        """Regenerate q-theta grid with current dimensions - to be overridden"""
        raise NotImplementedError("Subclasses must implement _regenerate_grid")
    
    def genes_to_struc_features(self, genevalues):
        """Convert genes to structural features - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement genes_to_struc_features")
    
    def get_feature_names(self):
        """Get feature names for XGBoost model - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement get_feature_names")
    
    def get_feature_titles(self):
        """Get titles for visualization - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement get_feature_titles")

#=============================================================================
# Hollow Tubes Model Configuration
#=============================================================================
class HollowTubesConfig(ModelConfig):
    """Configuration for Hollow Tubes model"""
    def __init__(self, input_shape=None):
        self.theta_min = 0
        self.theta_max = 180
        self.q_min_exp = -2.1
        self.q_max_exp = -0.9
        self.numgenes = 9  # Number of genes for hollow tubes
        super().__init__(input_shape)
    
    def setup_qtheta_grid(self):
        """Setup q-theta grid for Hollow Tubes"""
        # Default dimensions
        self.nq = 61 if self.input_shape is None else self.input_shape[0]
        self.ntheta = 61 if self.input_shape is None else self.input_shape[1]
        self._regenerate_grid()
    
    def _regenerate_grid(self):
        """Regenerate grid with current dimensions"""
        self.nqtheta = self.nq * self.ntheta
        self.theta_vals = np.linspace(self.theta_min, self.theta_max, self.ntheta)
        self.q_exp_vals = np.linspace(self.q_min_exp, self.q_max_exp, self.nq)
        self.q_theta_pairs = np.array([(q, theta) for q in self.q_exp_vals for theta in self.theta_vals])
        self.q_theta_pairs_3D = self.q_theta_pairs.reshape((self.nqtheta, 2, 1))
    
    def genes_to_struc_features(self, genevalues):
        """Convert genes to structural features for Hollow Tubes"""
        #Mean tube diameter
        dia = genevalues[:,0]*300+100 #Range from 100 to 400
        #Mean and Fractional SD of Eccentricity
        ecc_m = genevalues[:,1] #from 0 to 1
        ecc_sd = genevalues[:,2] #from 0 to 1
        #Mean Orientation Angle for the tubes
        orient = genevalues[:,3]*90 #from 0 to 90
        #exponent of Kappa
        kappaexp = genevalues[:,4]*10-5 #from -5 to 5
        #cone angle
        cone_angle = genevalues[:,5]*90 #from 0 to 90
        #Herding tubes diameter
        herd_dia = genevalues[:,6]*0.5 #from 0 to 0.5
        #Herding tubes length
        herd_len = genevalues[:,7] #from 0 to 1
        #Herding tubes number of extra nodes
        herd_extra_node = np.round(genevalues[:,8]*5) #Integers from 0 to 5
        struc_features = np.vstack((dia, ecc_m, ecc_sd, orient, kappaexp, cone_angle, herd_dia, herd_len, herd_extra_node))
        struc_features = struc_features.transpose()
        return struc_features
    
    def get_feature_names(self):
        """Get feature names for XGBoost model"""
        return ["Meandia", "MeanEcc", "FracSDEcc", "OrientAngle", "Kappa", "ConeAngle", 
                "HerdDia", "HerdLen", "HerdExtraNodes", "q_exp", "theta"]
    
    def get_feature_titles(self):
        """Get titles for visualization"""
        return ['mean diameter', 'mean ecc', 'frac std ecc', 'orient', 'kappa exp', 
                'cone angle', 'herd diameter', 'herd len', 'herd num extra nodes']

#=============================================================================
# Ellipsoids Model Configuration
#=============================================================================
class EllipsoidsConfig(ModelConfig):
    """Configuration for Ellipsoids model"""
    def __init__(self, input_shape=None):
        self.q_min = -2
        self.q_max = 3
        self.theta_min = 0
        self.theta_max = np.pi
        self.numgenes = 6  # Number of genes for ellipsoids
        super().__init__(input_shape)
    
    def setup_qtheta_grid(self):
        """Setup q-theta grid for Ellipsoids"""
        # Default dimensions
        self.nq = 126 if self.input_shape is None else self.input_shape[0]
        self.ntheta = 37 if self.input_shape is None else self.input_shape[1]
        self.numq = self.nq
        self.numtheta = self.ntheta
        self._regenerate_grid()
    
    def _regenerate_grid(self):
        """Regenerate grid with current dimensions"""
        self.nqtheta = self.nq * self.ntheta
        self.log_q = np.linspace(self.q_min, self.q_max, self.nq)
        self.theta = np.linspace(self.theta_min, self.theta_max, self.ntheta)
        self.q_theta_pairs = np.array([(q, theta) for q in self.log_q for theta in self.theta])
        self.q_theta_pairs_3D = self.q_theta_pairs.reshape((self.nqtheta, 2, 1))
    
    def genes_to_struc_features(self, genevalues):
        """Convert genes to structural features for Ellipsoids"""
        #Volume fraction
        phi = genevalues[:,0]/2
        #Radius
        logmuR = np.log(3) + genevalues[:,1]*np.log(10)
        logsigR = (1-np.abs((2*genevalues[:,1]-1)))*np.log(10)/6*genevalues[:,2]
        meanR = np.exp(logmuR+0.5*logsigR**2)
        sigmaR = np.sqrt((np.exp(logsigR**2)-1)*np.exp(2*logmuR+logsigR**2))
        #AspectRatios
        logmuG = (2*genevalues[:,3]-1)*np.log(10)
        logsigG = (1-np.abs((2*genevalues[:,3]-1)))*np.log(10)/3*genevalues[:,4]
        meanG = np.exp(logmuG+0.5*logsigG**2)
        sigmaG = np.sqrt((np.exp(logsigG**2)-1)*np.exp(2*logmuG+logsigG**2))
        #Kappa
        kappaexp = np.less_equal(genevalues[:,5],0.25)*((genevalues[:,5]/0.25)*(-1+10)-10) +\
                   np.greater(genevalues[:,5],0.25)*np.less_equal(genevalues[:,5],0.5)*(((genevalues[:,5]-0.25)/0.25)*(0+1)-1) +\
                   np.greater(genevalues[:,5],0.5)*np.less_equal(genevalues[:,5],0.75)*(((genevalues[:,5]-0.5)/0.25)*(1-0)) +\
                   np.greater(genevalues[:,5],0.75)*(((genevalues[:,5]-0.75)/0.25)*(10-1)+1)
        kappa = 10**kappaexp
        struc_features = np.vstack((meanR, sigmaR, meanG, sigmaG, kappa, phi))
        struc_features = struc_features.transpose()
        return struc_features
    
    def get_feature_names(self):
        """Get feature names for XGBoost model"""
        return ["MeanR", "StdR", "MeanG", "StdG", "Kappa", "Volume_Fraction", "log_q", "theta"]
    
    def get_feature_titles(self):
        """Get titles for visualization"""
        return ['mean radius', 'sigma radius', 'mean aspect ratio', 'sigma aspect ratio', 
                'kappa', 'volume fraction']

#=============================================================================
# Legacy global variables for backward compatibility (Hollow Tubes)
#=============================================================================
ntheta=61
nq=61
nqtheta=nq*ntheta
theta_min=0
theta_max=180
theta_vals = np.linspace(theta_min, theta_max, ntheta)
q_min_exp=-2.1
q_max_exp=-0.9
q_exp_vals = np.linspace(q_min_exp,q_max_exp,nq)
q_theta_pairs = np.array([(q, theta) for q in q_exp_vals for theta in theta_vals])
q_theta_pairs_3D=q_theta_pairs.reshape((nqtheta,2,1)) #Reshape to a 3D array

#=============================================================================
# Helper Functions
#=============================================================================
def get_model_config(model_type):
    """
    Factory function to get the appropriate model configuration.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('hollowTubes', 'Ellipsoids', or 'ellipsoids')
    
    Returns:
    --------
    ModelConfig instance for the specified model type
    """
    model_type_lower = model_type.lower()
    if model_type_lower in ['hollowtube', 'hollowtubes', 'hollow_tubes']:
        return HollowTubesConfig()
    elif model_type_lower in ['ellipsoid', 'ellipsoids']:
        return EllipsoidsConfig()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'hollowTubes', 'Ellipsoids'")

def load_model(model_type, models_dir='models'):
    """
    Load the XGBoost model for the specified model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('hollowTubes' or 'Ellipsoids')
    models_dir : str
        Directory containing the model files
    
    Returns:
    --------
    Loaded XGBoost model
    """
    if model_type.lower() in ['hollowtube', 'hollowtubes', 'hollow_tubes']:
        model_filename = 'hollowTubes_xgbModel.json'
    elif model_type.lower() in ['ellipsoid', 'ellipsoids']:
        model_filename = 'Ellipsoids_xgbModel.json'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    search_roots = [models_dir, os.path.join(os.path.dirname(__file__), '..', 'models')]
    # Fall back to original folder structure if available
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    search_roots.append(os.path.join(project_root, '01', 'models'))
    search_roots.append(os.path.join(project_root, 'models'))

    last_error = None
    for root in search_roots:
        candidate_path = os.path.join(root, model_filename)
        if os.path.exists(candidate_path):
            xgbmodel = xgb.Booster()
            xgbmodel.load_model(candidate_path)
            return xgbmodel
        last_error = candidate_path

    raise FileNotFoundError(f"Unable to locate model file '{model_filename}'. Last path checked: {last_error}")

#Function to Read Input File
def read_Iq(infilename,input_data_dir):
    '''
    function to read the input file
    '''
    file_path = os.path.join(input_data_dir, infilename)
    Iq_vals = np.genfromtxt(file_path,delimiter=',')
    return Iq_vals

#Function to Visualize scattering profile
def visualize_profile(profile,outfilename):
    if profile is None:
        return

    plt.figure(figsize = (10,10))
    plt.imshow(profile, cmap='inferno', vmin=np.min(0), vmax=np.max(6))
    plt.xticks(np.linspace(0,profile.shape[0],4),["0","60","120","180"])
    plt.yticks(np.linspace(0,profile.shape[1],5),["$10^{-0.9}$","$10^{-1.2}$","$10^{-1.5}$","$10^{-1.8}$","$10^{-2.9}$"])
    plt.xlabel("$\\theta (\\degree)$",fontsize=20)
    plt.ylabel("$q$",fontsize=20)
    if outfilename is not None:
        plt.savefig(outfilename)
    plt.show()

#Convert Genes to Structural Features (Legacy function for backward compatibility)
def genes_to_struc_features(genevalues, model_config=None):
    """
    Convert genes to structural features.
    If model_config is provided, uses the model-specific conversion.
    Otherwise, defaults to Hollow Tubes model for backward compatibility.
    """
    if model_config is not None:
        return model_config.genes_to_struc_features(genevalues)
    
    # Default: Hollow Tubes model (legacy behavior)
    #Mean tube diameter
    dia = genevalues[:,0]*300+100 #Range from 100 to 400
    #Mean and Fractional SD of Eccentricity
    ecc_m = genevalues[:,1] #from 0 to 1
    ecc_sd = genevalues[:,2] #from 0 to 1
    #Mean Orientation Angle for the tubes
    orient = genevalues[:,3]*90 #from 0 to 90
    #exponent of Kappa
    kappaexp = genevalues[:,4]*10-5 #from -5 to 5
    #cone angle
    cone_angle = genevalues[:,5]*90 #from 0 to 90
    #Herding tubes diameter
    herd_dia = genevalues[:,6]*0.5 #from 0 to 0.5
    #Herding tubes length
    herd_len = genevalues[:,7] #from 0 to 1
    #Herding tubes number of extra nodes
    herd_extra_node = np.round(genevalues[:,8]*5) #Integers from 0 to 5
    struc_features = np.vstack((dia, ecc_m, ecc_sd, orient, kappaexp, cone_angle, herd_dia, herd_len, herd_extra_node))
    struc_features=struc_features.transpose()
    return struc_features


#Convert Structural Features to xgb model input
def generate_xgbinput(struc_features, model_config=None):
    """
    Include q and theta with structural features.
    If model_config is provided, uses model-specific q-theta grid.
    Otherwise, defaults to legacy global variables for backward compatibility.
    """
    if model_config is not None:
        # Use model-specific configuration
        nqtheta_local = model_config.nqtheta
        q_theta_pairs_3D_local = model_config.q_theta_pairs_3D
    else:
        # Use global variables for backward compatibility
        nqtheta_local = nqtheta
        q_theta_pairs_3D_local = q_theta_pairs_3D
    
    shape_struc_features = struc_features.shape
    struc_features_reshaped = struc_features.transpose()
    struc_features_reshaped = struc_features_reshaped.reshape((1, shape_struc_features[1], shape_struc_features[0]))
    repeated_struc_features = np.repeat(struc_features_reshaped, repeats=nqtheta_local, axis=0)
    repeated_qtheta = np.repeat(q_theta_pairs_3D_local, repeats=shape_struc_features[0], axis=2)
    xgbinputs = np.hstack((repeated_struc_features, repeated_qtheta))
    return xgbinputs

#Use a single xgbinput to generate a single xgboutput
def generate_xgboutput(xgbinput, xgbmodel, model_config=None):
    """
    Generate XGBoost output from input features.
    If model_config is provided, uses model-specific feature names.
    Otherwise, defaults to Hollow Tubes feature names for backward compatibility.
    """
    if model_config is not None:
        feature_names = model_config.get_feature_names()
    else:
        # Default: Hollow Tubes feature names
        feature_names = ["Meandia", "MeanEcc", "FracSDEcc", "OrientAngle", "Kappa", "ConeAngle", 
                        "HerdDia", "HerdLen", "HerdExtraNodes", "q_exp", "theta"]
    
    dmatrix = xgb.DMatrix(xgbinput, feature_names=feature_names)
    xgboutput = xgbmodel.predict(dmatrix)
    return xgboutput

#Use entries from the GA Table to generate all the profiles of the current generation
def generateallprofiles(gatable, xgbmodel, model_config=None):
    """
    Generate all profiles for the current generation.
    If model_config is provided, uses model-specific configuration.
    Otherwise, defaults to legacy behavior for backward compatibility.
    """
    popsize = gatable.shape[0]
    numcols = gatable.shape[1]
    
    # Exclude the fitness column (last column)
    indscore = numcols - 1
    strucfeatures = genes_to_struc_features(gatable[:,0:indscore], model_config)
    
    xgbinputs = generate_xgbinput(strucfeatures, model_config)
    shape_xgbinputs = xgbinputs.shape
    
    # Get feature names from model_config or use default
    if model_config is not None:
        feature_names = model_config.get_feature_names()
        nq_local = model_config.nq
        ntheta_local = model_config.ntheta
    else:
        feature_names = ["Meandia", "MeanEcc", "FracSDEcc", "OrientAngle", "Kappa", "ConeAngle", 
                        "HerdDia", "HerdLen", "HerdExtraNodes", "q_exp", "theta"]
        nq_local = nq
        ntheta_local = ntheta
    
    xgbinputs = xgbinputs.transpose(2,0,1).reshape(popsize * shape_xgbinputs[0], shape_xgbinputs[1])
    dmatrix = xgb.DMatrix(xgbinputs, feature_names=feature_names)
    xgboutput = xgbmodel.predict(dmatrix)
    
    # Note: For Ellipsoids, both input data and model predictions are already in log space
    # Note: For Hollow tubes, both input data and model predictions are in real space
    # No transformation needed - fitness calculation compares values in their native space
    
    xgboutput = xgboutput.reshape((popsize, nq_local, ntheta_local))
    generated_profiles = xgboutput.transpose(1,2,0)
    return strucfeatures, generated_profiles
    
#Function to visualize distribution of structural features
def visualize_strucfeatures_dist(strucfeatures, outfilename, model_config=None):
    """
    Visualize distribution of structural features.
    If model_config is provided, uses model-specific feature titles.
    Otherwise, defaults to Hollow Tubes titles for backward compatibility.
    """
    if strucfeatures is None:
        return
    
    num_features = strucfeatures.shape[1]
    
    # Determine grid size based on number of features
    if num_features <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axs = plt.subplots(rows, cols, sharey=True, tight_layout=True)
    n_bins = 20
    
    # Get feature titles
    if model_config is not None:
        feature_titles = model_config.get_feature_titles()
    else:
        # Default: Hollow Tubes titles
        feature_titles = ['mean diameter', 'mean ecc', 'frac std ecc', 'orient', 'kappa exp', 
                         'cone angle', 'herd diameter', 'herd len', 'herd num extra nodes']
    
    # Plot histograms
    for i in range(num_features):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axs[col]
        else:
            ax = axs[row, col]
        ax.hist(strucfeatures[:,i], bins=n_bins)
        if i < len(feature_titles):
            ax.set_title(feature_titles[i])
    
    # Hide unused subplots
    for i in range(num_features, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            axs[col].set_visible(False)
        else:
            axs[row, col].set_visible(False)
    
    if outfilename is not None:
        plt.savefig(outfilename)
    plt.show()

# Calculate the fitness of the data1 with respect to data2
def calculate_fitness(data1, data2, use_log=False):
    """
    Calculate fitness using SSIM.
    
    Parameters:
    -----------
    data1 : np.array
        First data array (usually input data)
    data2 : np.array
        Second data array (usually generated profile)
    use_log : bool
        If True, assumes data is in real space and takes log before comparison.
        If False, compares data directly (assumes both are already in same space)
    """
    if use_log:
        # Take log of data (assumes data is in real space, convert to log space)
        data1 = np.log10(np.maximum(data1, 1e-10))  # Avoid log(0)
        data2 = np.log10(np.maximum(data2, 1e-10))
    
    datarange = np.max([np.max(data1),np.max(data2)]) - np.min([np.min(data1),np.min(data2)])
    try:
        score = ssim(data1, data2, data_range=datarange)
        #score = r2_score(data1, data2)
        return score
    except Exception as e:
        print("Error calculating SSIM:", str(e))
        return None

# Update all the fitness for the gatable
def updateallfitnesses(gatable, profiles, inputdata, use_log=False):
    """
    Update fitness scores for all individuals in GA table.
    
    Parameters:
    -----------
    gatable : np.array
        GA table with genes and fitness column
    profiles : np.array
        Generated profiles
    inputdata : np.array
        Input data to compare against
    use_log : bool
        If True, converts to log space before fitness calculation
    """
    popsize = gatable.shape[0]
    indscore = gatable.shape[1] - 1
    for n in range(popsize):
         gatable[n, indscore] = calculate_fitness(inputdata, profiles[:,:,n], use_log=use_log)
    return gatable

# Plot the fitness as a function of the number of generations
def plot_fitness(fitness_scores, ax, outfilename):
    line1 = ax.errorbar(fitness_scores[:,0],fitness_scores[:,1],yerr=fitness_scores[:,2], fmt='o', color='red', linewidth=1)
    line2, = ax.plot(fitness_scores[:,0],fitness_scores[:,3],'k-',label='best fitness',linewidth=2)
    line3, = ax.plot(fitness_scores[:,0],fitness_scores[:,4],'b-',label='worst fitness',linewidth=2)
    #line3, = ax.plot(fitness_scores[:,0],fitness_scores[:,3])
    # setting title
    plt.autoscale(enable=True,axis='both')
    #ax.set(xlim=(0,numgens),ylim=(0.7,0.9))
    plt.title("Fitness vs Generation", fontsize=20)
    plt.xlabel("Generation", fontsize=20)
    plt.ylabel("Fitness Value (SSIM)", fontsize=20)
    if outfilename is not None:
        plt.savefig(outfilename)
    plt.show()

#GA operations (mating and crossover)
def generate_children(parents, popsize, numgenes):
    size_parents = parents.shape
    numparents = size_parents[0]
    numchildren = popsize - numparents
    if numchildren % 2 !=0:
        print('numchildren must be even!')
        return None
    numpairs = int(numchildren/2)
    numcols = size_parents[1]
    #Using rank weighting for parent selection
    randnumbersparent = np.random.rand(numchildren)
    #each two consecutive rows mate
    parentindices = np.int64(np.floor((2*numparents+1-np.sqrt(4*numparents*(1+numparents)*(1-randnumbersparent)+1))/2))
    children = parents[parentindices,:]
    # perform crossover
    crossoverpoint = np.random.rand(numpairs)*numgenes
    crossoverindex = np.int64(np.floor(crossoverpoint))
    crossovervalue = crossoverpoint - crossoverindex
    for n in range(numpairs):
        originalchild1 = children[2*n,:]
        originalchild2 = children[2*n+1,:]
        ind=crossoverindex[n]
        val=crossovervalue[n]
        newchild1 = np.hstack((originalchild1[0:ind],originalchild2[ind:]))
        newchild2 = np.hstack((originalchild2[0:ind],originalchild1[ind:]))
        newchild1[ind]= originalchild1[ind]*val+originalchild2[ind]*(1-val)
        newchild2[ind]= originalchild2[ind]*val+originalchild1[ind]*(1-val)
        newchild1[ind]=np.maximum(np.minimum(newchild1[ind],1),0)
        newchild2[ind]=np.maximum(np.minimum(newchild2[ind],1),0)
        #np.clip(newchild1[ind], 0, 1, out=newchild1[ind])
        #np.clip(newchild2[ind], 0, 1, out=newchild2[ind])
        children[2*n,:]=newchild1
        children[2*n+1,:]=newchild2
    return children

#GA operations (mutations)
def applymutations(gatable,numelites,mutationrate):
    shape_gatable = gatable.shape
    mutationhalfstepsize = 0.15
    mutationflag = np.less_equal(np.random.rand(shape_gatable[0],shape_gatable[1]),mutationrate)
    mutationvalues = np.random.uniform(-mutationhalfstepsize,mutationhalfstepsize,(shape_gatable[0],shape_gatable[1]))*mutationflag
    mutationvalues[0:numelites,:] = 0 #elite individuals are not mutated
    gatable = gatable + mutationvalues
    np.clip(gatable, 0, 1, out=gatable)    
    return gatable
