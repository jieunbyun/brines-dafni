import json
import pandas as pd
import numpy as np
import xarray as xr
import pickle
import os
import zipfile
import glob
import re
import shutil

import matplotlib
matplotlib.use('Agg')

from scipy.stats import multivariate_normal
import copy
from copulas.multivariate import GaussianMultivariate

from pyomo.environ import *
from pyomo.dae import *
import pandas as pd

class RenewableEnergyUQ:
    def __init__(self, data_fpaths, start_date, end_date, gPATHI, gPATHO):
        """        
        Parameters:
        - data_fpaths (str): Dictionary of file paths
        - start_date (str): Start date for processing data.
        - end_date (str): End date for processing data.
        - gPATHI (str): Input path for the data.
        - gPATHO (str): Output path for the data.
        """

        self.data_fpaths = data_fpaths
        self.start_date = start_date
        self.end_date = end_date

        # Load JSON datasets
        self.buses = self._load_and_sort_json("buses", sort_by="name")
        self.lines = self._load_and_sort_json("transmission_lines", sort_by="bus0")
        self.generators_wind = self._load_and_sort_json("wind_generators", sort_by="bus")
        self.generators_pv = self._load_and_sort_json("solar_generators", sort_by="bus")

        # Load city name mapping
        city_mapping_path = self.data_fpaths["city_name_mapping"]
        with open(city_mapping_path, 'r') as f:
            self.city_fname_to_codename = json.load(f)

        # Load and process NetCDF files
        self.dfs_tas = self.process_netcdf_files('tas')
        self.save_dict_to_hdf5(self.dfs_tas, os.path.join(gPATHO, "temperature_data.h5"))

        self.dfs_sfc_wind = self.process_netcdf_files('sfcWind')
        self.save_dict_to_hdf5(self.dfs_sfc_wind, os.path.join(gPATHO, "wind_speed_data.h5"))

        self.dfs_solar = self.process_netcdf_files('rss')
        self.save_dict_to_hdf5(self.dfs_solar, os.path.join(gPATHO, "solar_data.h5"))


    def _load_and_sort_json(self, key, sort_by):
        """
        Loads a JSON file into a DataFrame, sorts it by a specified column, and returns the sorted DataFrame.

        Parameters:
        - key (str): Key to retrieve the file path from `self.data_fpaths`.
        - sort_by (str): Column name to sort the DataFrame.

        Returns:
        - pd.DataFrame: Sorted DataFrame, or an empty DataFrame if file is missing.
        """
        file_path = self.data_fpaths.get(key, "")

        if not file_path or not os.path.exists(file_path):
            print(f"Warning: {key} file not found at {file_path}. Returning an empty DataFrame.")
            return pd.DataFrame()  # Return an empty DataFrame if the file is missing

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Ensure data is a list of dictionaries
            if isinstance(data, dict):
                print(f"Warning: Expected a list of records in {file_path}, got a dictionary. Converting to list.")
                data = [data]  # Wrap dictionary in a list

            df = pd.DataFrame(data)

            # Ensure the sorting column exists
            if sort_by not in df.columns:
                print(f"Warning: Column '{sort_by}' not found in {file_path}. Returning unsorted data.")
                return df  # Return unsorted DataFrame

            return df.sort_values(by=sort_by).reset_index(drop=True)

        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON file {file_path}. {e}")
            return pd.DataFrame()  # Return an empty DataFrame if JSON is invalid

    def process_netcdf_files(self, variable):
        """
        Processes NetCDF files for a given variable from a specified ZIP archive,
        extracting only relevant folders (e.g., ENS_01, ENS_04).

        Parameters:
        - variable (str): The NetCDF variable to extract (e.g., 'tas', 'sfcWind', 'rss').

        Returns:
        - dict: Dictionary where keys are city names (codenames) and values are processed DataFrames.
        """
        # Select the correct dataset key
        file_mapping = {
            "tas": self.data_fpaths["temperature_data"],
            "sfcWind": self.data_fpaths["wind_speed_data"],
            "rss": self.data_fpaths["solar_data"]
        }

        zip_file_path = file_mapping.get(variable, "")

        if not zip_file_path:
            print(f"Error: No file mapping found for variable '{variable}'.")
            return {}

        extracted_folder = "/tmp/extracted_netcdf"  # Temporary folder for extracted NetCDF files
        os.makedirs(extracted_folder, exist_ok=True)

        with zipfile.ZipFile(zip_file_path[0], 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)

        # Extract all ZIP files
        netcdf_files = glob.glob(os.path.join(extracted_folder, "**", "*.nc"), recursive=True)

        if not netcdf_files:
            print(f"Error: No NetCDF files found after extracting ZIPs for {variable}.")
            return {}

        # Process NetCDF files
        dfs_data = {}
        base_time = pd.Timestamp("1970-01-01")

        for file_path in netcdf_files:
            print(f"Processing: {file_path}")

            file_name = os.path.basename(file_path)

            # Extract city name dynamically
            file_city = None
            for city_name in self.city_fname_to_codename.keys():
                if city_name in file_name:
                    file_city = city_name
                    break

            if file_city is not None:
                code_city = self.city_fname_to_codename[file_city]

                with xr.open_dataset(file_path, decode_times=False) as ds:
                    df = ds.to_dataframe().reset_index()

                # Convert time variable
                df['time'] = base_time + pd.to_timedelta(df['time'], unit='h')

                # Filter data within the specified time range
                df_filtered = df[(df['time'] >= self.start_date) & (df['time'] <= self.end_date)]

                # Store the processed DataFrame
                df_processed = (
                    df_filtered[['time', variable]]
                    .assign(
                        year=lambda x: pd.to_datetime(x['time']).dt.year,
                        month=lambda x: pd.to_datetime(x['time']).dt.month,
                        day=lambda x: pd.to_datetime(x['time']).dt.day
                    )
                    .drop(columns=['time'])
                )

                if code_city in dfs_data:
                    dfs_data[code_city] = pd.concat([dfs_data[code_city], df_processed], ignore_index=True).drop_duplicates()
                else:
                    dfs_data[code_city] = df_processed

            else:
                print(f"Warning: Could not extract city name from {file_path}.")

        shutil.rmtree(extracted_folder)
        print(f"Deleted folder: {extracted_folder}")
        
        return dfs_data
    
    @staticmethod
    def save_dict_to_hdf5(data_dict, file_path):
        """
        Saves a dictionary of DataFrames into a single HDF5 file.

        Parameters:
        - data_dict (dict): Dictionary where keys are dataset names and values are DataFrames.
        - file_path (str): Path to save the HDF5 file.
        """
        with pd.HDFStore(file_path, mode='w') as store:
            for key, df in data_dict.items():
                store.put(key, df, format='table', data_columns=True)  # Use table format for fast querying
                print(f"Saved: {key} ({df.shape})")
    
    def sample_temperature(self, n_samples, rand_seed=1, plot_codename_pair = None):
        np.random.seed(rand_seed)
        
        cities = list(self.dfs_tas.keys())
        min_length = min(len(data) for data in self.dfs_tas.values())
        
        if min_length == 0:
            raise ValueError("Insufficient data for the given year and period across all cities.")
        
        temp_data = np.array([data[:min_length]['tas'].values for data in self.dfs_tas.values()]).T
        
        mean_vector = np.mean(temp_data, axis=0)
        cov_matrix = np.cov(temp_data, rowvar=False)
        sampled_data = multivariate_normal.rvs(mean=mean_vector, cov=cov_matrix, size=n_samples)
        
        sampled_results = {city: sampled_data[:, idx] for idx, city in enumerate(cities)}

        if plot_codename_pair:
            self.plot_sampled_vs_actual(self.dfs_tas, sampled_results,
                                            'tas', plot_codename_pair[0], plot_codename_pair[1])

        return pd.DataFrame(sampled_results)
    
    def sample_net_demand(self, samples_temp_df, rand_seed=1):
        np.random.seed(rand_seed)

        poly_coeffs = [52743.8362, -357.4178, -51.8631, 0.1017, 0.0655]
        total_demand = self.buses['demand'].sum()
        demand_ratio = self.buses.set_index('name')['demand'] / total_demand
        
        def polynomial_regression(temp):
            max_demand = (
                poly_coeffs[0] +
                poly_coeffs[1] * -2.5 +
                poly_coeffs[2] * (-2.5)**2 +
                poly_coeffs[3] * (-2.5)**3 +
                poly_coeffs[4] * (-2.5)**4
            )
            estimated_demand = (
                poly_coeffs[0] +
                poly_coeffs[1] * temp +
                poly_coeffs[2] * temp**2 +
                poly_coeffs[3] * temp**3 +
                poly_coeffs[4] * temp**4
            )
            return min(estimated_demand, max_demand)
        
        df_demand_predicted = samples_temp_df.applymap(polynomial_regression)
        noise = np.random.normal(0, 5012.1401, size=samples_temp_df.shape)
        df_demand_predicted += noise
        df_net_demand = df_demand_predicted * demand_ratio.reindex(df_demand_predicted.columns).values
        return df_net_demand
    
    def sample_wind_speed(self, n_samples, rand_seed=1, plot_codename_pair = None):
        np.random.seed(rand_seed)

        cities = list(self.dfs_sfc_wind.keys())
        wind_data = []
        
        for city in cities:
            df = copy.deepcopy(self.dfs_sfc_wind[city])
            monthly_data = df['sfcWind'].dropna()
            wind_data.append(np.log(monthly_data.values))  # Apply log transformation
        
        min_length = min(len(data) for data in wind_data)
        wind_data = np.array([data[:min_length] for data in wind_data]).T
        
        mean_vector = np.mean(wind_data, axis=0)
        cov_matrix = np.cov(wind_data, rowvar=False)
        sampled_log_data = multivariate_normal.rvs(mean=mean_vector, cov=cov_matrix, size=n_samples)
        
        sampled_data = np.exp(sampled_log_data)
        sampled_results = {city: sampled_data[:, idx] for idx, city in enumerate(cities)}

        if plot_codename_pair:
            self.plot_sampled_vs_actual(self.dfs_sfc_wind, sampled_results,
                                            'sfcWind', plot_codename_pair[0], plot_codename_pair[1])

        return pd.DataFrame(sampled_results)
    
    def sample_solar_radiation(self, n_samples, rand_seed=1, plot_codename_pair = None):
        np.random.seed(rand_seed)

        cities = list(self.dfs_solar.keys())
        monthly_data = pd.DataFrame({
            city: self.dfs_solar[city]['rss'].values / 300
            for city in cities
        }).dropna(axis=1, how='all')
        
        copula = GaussianMultivariate()
        copula.fit(monthly_data)
        samples = copula.sample(n_samples) * 300
        samples = np.clip(samples, 0, None)

        sampled_results = {city: samples[city].to_list() for city in cities}

        if plot_codename_pair:
            self.plot_sampled_vs_actual(self.dfs_solar, sampled_results,
                                            'rss', plot_codename_pair[0], plot_codename_pair[1])

        return pd.DataFrame(sampled_results)
    
    def optimise_energy_system(self, sampled_wind_df, sampled_solar_df, sampled_demand_df):
        optim_results = {
            "status": [], "loadshed": []
        }
        
        n_samples = sampled_demand_df.shape[0]
        for i in range(n_samples):
            print(f"Sample {i + 1}/{n_samples}")
            sampled_nd = sampled_demand_df.iloc[i]
            sampled_wspd = sampled_wind_df.iloc[i]
            sampled_sr = sampled_solar_df.iloc[i]
            
            nodes = self.buses['name'].tolist()
            lines = list(zip(self.lines['bus0'], self.lines['bus1']))
            wind_generators = list(zip(self.generators_wind['name'], self.generators_wind['bus']))
            pv_generators = list(zip(self.generators_pv['name'], self.generators_pv['bus']))

            wind_limits = {row['name']: row['p_nom'] for _, row in self.generators_wind.iterrows()}
            pv_limits = {row['name']: row['p_nom'] for _, row in self.generators_pv.iterrows()}
            
            def create_model():
                model = ConcreteModel()
                model.P_G_wind = Var(wind_generators, domain=NonNegativeReals)
                model.P_G_pv = Var(pv_generators, domain=NonNegativeReals)
                model.theta = Var(nodes, domain=Reals)
                model.P_L = Var(lines, domain=Reals)
                model.S = Var(nodes, domain=NonNegativeReals)
                
                node_demand_dict = {node: sampled_nd[node] for node in nodes}
                wind_limit_dict = {(gen, bus): wind_limits[gen] * self.wind_power_ratio(sampled_wspd.get(bus, 0))
                                   for gen, bus in wind_generators}
                pv_limit_dict = {(gen, bus): pv_limits[gen] * (sampled_sr.get(bus, 0) / 300) for gen, bus in pv_generators}
                
                def power_balance_rule(model, node):
                    incoming = sum(model.P_L[(n1, n2)] for n1, n2 in lines if n2 == node)
                    outgoing = sum(model.P_L[(n1, n2)] for n1, n2 in lines if n1 == node)
                    generation_wind = sum(model.P_G_wind[(gen, bus)] for gen, bus in wind_generators if bus == node)
                    generation_pv = sum(model.P_G_pv[(gen, bus)] for gen, bus in pv_generators if bus == node)
                    return incoming - outgoing + generation_wind + generation_pv + model.S[node] == node_demand_dict[node]
                
                model.eqPowerBalance = Constraint(nodes, rule=power_balance_rule)
                
                def loadshed_limit_rule(model, node):
                    return model.S[node] <= node_demand_dict[node]
                model.ineqLoadshedLimit = Constraint(nodes, rule=loadshed_limit_rule)

                line_resistance = {(row['bus0'], row['bus1']): row['x'] for _, row in self.lines.iterrows()}
                def line_flow_rule(model, n1, n2):
                    return model.P_L[(n1, n2)] == line_resistance[(n1, n2)] * (model.theta[n1] - model.theta[n2])
                model.eqLineFlow = Constraint(lines, rule=line_flow_rule)

                line_capacity = {(row['bus0'], row['bus1']): row['s_nom'] for _, row in self.lines.iterrows()}
                def line_capacity_rule(model, n1, n2):
                    return (-line_capacity[(n1, n2)], model.P_L[(n1, n2)], line_capacity[(n1, n2)])
                model.boundP_L = Constraint(lines, rule=line_capacity_rule)

                def wind_capacity_rule(model, gen, bus):
                    return model.P_G_wind[(gen, bus)] <= wind_limit_dict.get((gen, bus), wind_limits[gen])

                def pv_capacity_rule(model, gen, bus):
                    return model.P_G_pv[(gen, bus)] <= pv_limit_dict.get((gen, bus), pv_limits[gen])

                model.windCapacity = Constraint(wind_generators, rule=wind_capacity_rule)
                model.pvCapacity = Constraint(pv_generators, rule=pv_capacity_rule)
                
                ref_bus = nodes[0]
                def reference_bus_rule(model):
                    return model.theta[ref_bus] == 0
                model.eqReferenceBus = Constraint(rule=reference_bus_rule)
                
                model.cost = Objective(expr=sum(model.S[node] for node in nodes), sense=minimize)

                return model
            
            model = create_model()
            solver = SolverFactory('appsi_highs')
            print(solver.available())
            solver.options['max_iter'] = 10000
            solver.options['tol'] = 1e-2
            solver.options['mu_strategy'] = 'adaptive'
            result = solver.solve(model)
            
            optim_results["loadshed"].append({node: model.S[node].value for node in nodes})
            optim_results["status"].append(result.solver.status == SolverStatus.ok)
            
            if result.solver.status == SolverStatus.ok:
                print("Optimisation successful.")
            else:
                print("Optimisation failed or infeasible.")
        
        return optim_results

    @staticmethod
    def save_loadshed_as_parquet(optim_results, filepath="loadshed.parquet"):
        city_names = list(optim_results["loadshed"][0].keys())
        loadshed_dict = {"optim_status": optim_results["status"]}
        loadshed_dict.update(
            {"total": [sum(optim_results["loadshed"][i].values()) for i in range(len(optim_results["loadshed"]))]}
        )
        loadshed_dict.update(
            {city: [optim_results["loadshed"][i][city] for i in range(len(optim_results["loadshed"]))]
            for city in city_names}
        )
        df = pd.DataFrame(loadshed_dict)
        df.to_parquet(filepath)
        print(f"Optimisation results saved as {filepath}")

    @staticmethod
    def wind_power_ratio(wind_speed, cut_in=3, rated=12, cut_out=25):
        if wind_speed <= cut_in or wind_speed >= cut_out:
            return 0
        elif cut_in <= wind_speed <= rated:
            return ((wind_speed - cut_in) / (rated - cut_in))**3
        else:
            return 1
    




