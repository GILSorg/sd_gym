import pysd
import logging
from datetime import datetime
import os

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'model_info_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_all_variables(model):
    """Get all variables in the model"""
    try:
        # Get components directly from the model
        variables = []
        
        # Get variables from different components
        variables.extend(model.components._stocknames)  # Stock variables
        variables.extend(model.components._flownames)   # Flow variables
        variables.extend(model.components._auxnames)    # Auxiliary variables
        
        # Get documentation for each variable
        var_info = {}
        for var in variables:
            try:
                doc_string = model.components._namespace[var]['doc']
                var_info[var] = doc_string if doc_string else "No documentation available"
            except:
                var_info[var] = "No documentation available"
        
        return var_info
        
    except Exception as e:
        logging.error(f"Error getting variables: {e}")
        return None

def get_constants(model):
    """Get all constant variables that can be controlled"""
    try:
        # Get constants from components
        constants = []
        for var in model.components._namespace:
            if model.components._namespace[var].get('type') == 'constant':
                constants.append(var)
        return constants
    except Exception as e:
        logging.error(f"Error getting constants: {e}")
        return None

def get_stocks(model):
    """Get all stock variables (state variables)"""
    try:
        # Get stocks from model namespace
        stocks = []
        for var in model.components._namespace:
            if model.components._namespace[var].get('type') == 'stock':
                stocks.append(var)
        return stocks
    except Exception as e:
        logging.error(f"Error getting stocks: {e}")
        return None

def analyze_model(model_path):
    """Analyze a PySD model and print its components"""
    logger = setup_logging()
    
    try:
        logger.info(f"Loading model from {model_path}")
        model = pysd.read_xmile(model_path)
        
        logger.info("\n=== All Model Variables ===")
        all_vars = get_all_variables(model)
        if all_vars is not None:
            logger.info(all_vars)
        
        logger.info("\n=== Constants (Potential Actionable Variables) ===")
        constants = get_constants(model)
        if constants is not None:
            logger.info(constants)
        
        logger.info("\n=== Stocks (State Variables) ===")
        stocks = get_stocks(model)
        if stocks is not None:
            logger.info(stocks)
            
        return all_vars, constants, stocks
        
    except Exception as e:
        logger.error(f"Error analyzing model: {e}")
        return None, None, None

if __name__ == "__main__":
    MODEL_PATH = '/tmp/electric_vehicles_norway.stmx'
    all_vars, constants, stocks = analyze_model(MODEL_PATH)