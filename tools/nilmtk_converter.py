#!/usr/bin/env python3
"""
NILMTK Converter for UK-DALE Dataset
Converts HDF5 format to pandas binary files with 6-second aggregated power data
"""

import argparse
import os
import re

import pandas as pd
from nilmtk import DataSet


class NILMTKConverter:
    """
    Converter class for NILMTK datasets to pandas binary format
    """

    def __init__(
        self,
        input_path,
        output_dir="./output",
        max_records=None,
        target_buildings=None,
        output_format="pickle",
        target_devices=None,
        strip_nan=False,
    ):
        """
        Initialize the converter

        Args:
            input_path (str): Path to the HDF5 dataset file
            output_dir (str): Output directory for binary files
            max_records (int): Maximum number of records per file (None for no limit)
            target_buildings (list): List of building numbers to export (None for all)
            output_format (str): Output format - 'pickle', 'parquet', or 'feather'
            target_devices (list): List of target device names (normalized) to export (None for all)
            strip_nan (bool): Whether to strip NaN rows from both sides
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.max_records = max_records
        self.target_buildings = target_buildings
        self.output_format = output_format.lower()
        self.target_devices = set(target_devices) if target_devices else None
        self.strip_nan = strip_nan

        # Validate output format
        if self.output_format not in ["pickle", "parquet", "feather"]:
            raise ValueError("output_format must be 'pickle', 'parquet', or 'feather'")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the dataset
        print(f"Loading dataset from: {self.input_path}")
        self.dataset = DataSet(self.input_path)

    def normalize_appliance_name(self, name):
        """
        Normalize appliance names to lowercase with underscores

        Args:
            name (str): Original appliance name

        Returns:
            str: Normalized appliance name (without _power suffix)
        """
        if name is None:
            return "unknown"

        # Convert to lowercase and replace spaces/special chars with underscores
        normalized = re.sub(r"[^a-zA-Z0-9]", "_", str(name).lower())
        # Remove multiple consecutive underscores
        normalized = re.sub(r"_+", "_", normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip("_")

        return normalized if normalized else "unknown"

    def get_building_info(self):
        """
        Get information about available buildings in the dataset

        Returns:
            dict: Building information
        """
        building_info = {}
        for building_num in self.dataset.buildings:
            building = self.dataset.buildings[building_num]
            meters = list(building.elec.meters)
            building_info[building_num] = {"num_meters": len(meters), "meters": meters}
        return building_info

    def generator_to_dataframe(self, generator, column_name):
        """
        Convert NILMTK generator to pandas DataFrame

        Args:
            generator: NILMTK power series generator
            column_name (str): Name for the power column

        Returns:
            pd.DataFrame: DataFrame with power data
        """
        data_list = []
        for chunk in generator:
            if chunk is not None and not chunk.empty:
                # Handle both Series and DataFrame
                if isinstance(chunk, pd.Series):
                    # Convert Series to DataFrame with specified column name
                    chunk_df = chunk.to_frame(name=column_name)
                elif isinstance(chunk, pd.DataFrame):
                    # Rename DataFrame columns
                    chunk_df = chunk.copy()
                    if len(chunk_df.columns) > 0:
                        chunk_df.columns = [column_name]
                else:
                    continue

                data_list.append(chunk_df)

        if data_list:
            return pd.concat(data_list, axis=0)
        else:
            return pd.DataFrame()

    def save_dataframe(self, df: pd.DataFrame, filepath: str) -> str:
        """
        Save DataFrame in the specified binary format

        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Output file path (without extension)
        """
        if self.output_format == "pickle":
            output_file = f"{filepath}.pkl"
            df.to_pickle(output_file)
        elif self.output_format == "parquet":
            output_file = f"{filepath}.parquet"
            df.to_parquet(output_file, index=True)
        elif self.output_format == "feather":
            output_file = f"{filepath}.feather"
            # Feather requires reset_index for datetime index
            df_reset = df.reset_index()
            df_reset.to_feather(output_file)

        return output_file

    def strip_nan_rows(self, df):
        """
        Strip NaN rows from both sides of the DataFrame

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with NaN rows stripped from both sides
        """
        if df.empty:
            return df

        # Find first and last rows that are not all NaN
        valid = ~df["mains"].isnull()

        if not valid.any():
            # All rows are NaN
            return pd.DataFrame(columns=df.columns, index=df.index[:0])

        first_valid = valid.idxmax()
        last_valid = valid[::-1].idxmax()

        # Check if there are NaN rows between first and last valid rows
        nan_rows_count = (~valid.loc[first_valid:last_valid]).sum()
        if nan_rows_count > 0:
            print(
                f"  Warning: {nan_rows_count} rows with NaN mains found between valid data points"
            )

        return df.loc[first_valid:last_valid]

    def export_building_to_binary(self, building_num):
        """
        Export a single building's data to binary format

        Args:
            building_num (int): Building number to export
        """
        print(f"\nProcessing Building {building_num}...")

        building = self.dataset.buildings[building_num]

        # Get mains data (single ElecMeter)
        mains = building.elec.mains()

        # Get mains power data
        print("  Getting mains data...")
        try:
            mains_generator = mains.power_series(sample_period=6)
            mains_df = self.generator_to_dataframe(mains_generator, "mains")

            if mains_df.empty:
                print(f"  Warning: No mains data found for building {building_num}")
                return

            print(f"  Mains data shape: {mains_df.shape}")

        except Exception as e:
            print(f"  Error getting mains data: {e}")
            return

        # Start with mains data
        combined_df = mains_df.copy()

        # Get individual appliance data
        print("  Processing individual appliances...")
        submeters = building.elec.submeters()

        for meter in submeters.meters:
            try:
                # Get appliance name
                appliance_name = "unknown"
                if hasattr(meter, "appliances") and meter.appliances:
                    appliance = meter.appliances[0]
                    if hasattr(appliance, "metadata") and "type" in appliance.metadata:
                        appliance_name = appliance.metadata["type"]

                # Normalize appliance name
                normalized_name = self.normalize_appliance_name(appliance_name)

                # Check if this device should be included
                if self.target_devices and normalized_name not in self.target_devices:
                    continue

                # Get power data for this meter
                meter_generator = meter.power_series(sample_period=6)

                # Use meter instance number if appliance name conflicts
                column_name = normalized_name
                if column_name in combined_df.columns:
                    column_name = f"{normalized_name}_{meter.instance()}"

                meter_df = self.generator_to_dataframe(meter_generator, column_name)

                if not meter_df.empty:
                    # Merge with combined data
                    combined_df = combined_df.join(meter_df, how="outer")
                    print(
                        f"    Added: {column_name} (shape: {meter_df.shape} / NaN: {meter_df.isnull().sum().sum()})"
                    )

            except Exception as e:
                print(f"    Warning: Could not process meter {meter.instance()}: {e}")
                continue

        if combined_df.empty:
            print(f"  Warning: No data found for building {building_num}")
            return

        # Remove rows with all NaN values
        combined_df = combined_df.dropna(how="all")

        # Strip NaN rows from both sides if requested
        if self.strip_nan:
            original_length = len(combined_df)
            combined_df = self.strip_nan_rows(combined_df)
            print(
                f"  Stripped NaN rows: {original_length} -> {len(combined_df)} records"
            )

        # Apply record limit if specified
        if self.max_records and len(combined_df) > self.max_records:
            print(
                f"  Limiting to {self.max_records} records (original: {len(combined_df)})"
            )
            combined_df = combined_df.head(self.max_records)

        # Save to binary format
        filepath = os.path.join(self.output_dir, f"building_{building_num}")
        output_file = self.save_dataframe(combined_df, filepath)

        print(f"  Exported {len(combined_df)} records to: {output_file}")
        print(f"  Columns: {list(combined_df.columns)}")
        print(f"  Data types: {combined_df.dtypes.value_counts().to_dict()}")
        print(
            f"  Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        )

    def convert(self):
        """
        Convert the dataset to binary files
        """
        print(f"Starting NILMTK to {self.output_format.upper()} conversion...")

        # Get building information
        building_info = self.get_building_info()
        print(f"\nDataset contains {len(building_info)} buildings")

        # Show target devices if specified
        if self.target_devices:
            print(f"Target devices: {sorted(self.target_devices)}")

        # Determine which buildings to process
        buildings_to_process = (
            self.target_buildings
            if self.target_buildings
            else list(building_info.keys())
        )

        print(f"Processing buildings: {buildings_to_process}")

        # Process each building
        for building_num in buildings_to_process:
            if building_num not in building_info:
                print(f"Warning: Building {building_num} not found in dataset")
                continue

            try:
                self.export_building_to_binary(building_num)
            except Exception as e:
                print(f"Error processing building {building_num}: {e}")
                continue

        print(f"\nConversion completed! Files saved to: {self.output_dir}")
        print(f"Format: {self.output_format.upper()}")

    def get_all_appliances(self):
        """
        Get all appliances from all buildings in the dataset

        Returns:
            dict: Dictionary with building numbers as keys and appliance lists as values
        """
        all_appliances = {}

        for building_num in self.dataset.buildings:
            building = self.dataset.buildings[building_num]
            appliances = set()

            # Add submeters
            submeters = building.elec.submeters()
            for meter in submeters.meters:
                try:
                    # Get appliance name
                    appliance_name = "unknown"
                    if hasattr(meter, "appliances") and meter.appliances:
                        appliance = meter.appliances[0]
                        if (
                            hasattr(appliance, "metadata")
                            and "type" in appliance.metadata
                        ):
                            appliance_name = appliance.metadata["type"]

                    # Normalize appliance name
                    normalized_name = self.normalize_appliance_name(appliance_name)
                    appliances.add(normalized_name)

                except Exception:
                    continue
            if "mains" in appliances:
                appliances.remove("mains")
            all_appliances[building_num] = sorted(list(appliances))

        return all_appliances

    def list_all_appliances(self, filter: list):
        """
        List all appliances in the dataset and exit
        """
        print("Listing all appliances in the dataset...")

        all_appliances = self.get_all_appliances()

        # Collect unique appliances across all buildings
        unique_appliances = set()
        for building_num, appliances in all_appliances.items():
            if filter:
                for target in filter:
                    for appliance in appliances:
                        if target in appliance:
                            unique_appliances.add(appliance)
            else:
                unique_appliances.update(appliances)

        print(
            f"\nFound {len(unique_appliances)} unique appliances across {len(all_appliances)} buildings:"
        )
        print("=" * 60)

        for appliance in sorted(unique_appliances):
            # Count in how many buildings this appliance appears
            building_count = sum(
                1 for appliances in all_appliances.values() if appliance in appliances
            )
            print(
                f"  {appliance:<30} (in {building_count}/{len(all_appliances)} buildings)"
            )

        print("\nPer-building breakdown:")
        print("=" * 60)

        for building_num in sorted(all_appliances.keys()):
            appliances = [
                app for app in all_appliances[building_num] if app in unique_appliances
            ]
            if not appliances:
                print(f"\nBuilding {building_num} has no appliance")
                continue
            print(f"\nBuilding {building_num} ({len(appliances)} appliances):")
            for appliance in appliances:
                print(f"  - {appliance}")


def main():
    """
    Main function with command line argument parsing
    """
    parser = argparse.ArgumentParser(
        description="Convert NILMTK HDF5 dataset to pandas binary files"
    )

    parser.add_argument(
        "--input",
        "-i",
        default="/home/user/workspaces/dataset/ukdale-raw/ukdale.h5",
        help="Path to input HDF5 file",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="./convert_output",
        help="Output directory for binary files (default: ./convert_output)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["pickle", "parquet", "feather"],
        default="feather",
        help="Output format (default: feather)",
    )

    parser.add_argument(
        "--max-records",
        "-m",
        type=int,
        default=None,
        help="Maximum number of records per file (default: no limit)",
    )

    parser.add_argument(
        "--buildings",
        "-b",
        nargs="+",
        type=int,
        default=None,
        help="Target building numbers to export (default: all buildings)",
    )

    parser.add_argument(
        "--devices",
        "-d",
        nargs="+",
        type=str,
        default=None,
        help="Target device names (normalized) to export (default: all devices)",
    )

    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip NaN rows from both sides of the data",
    )

    parser.add_argument(
        "--info", action="store_true", help="Show dataset information and exit"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all appliances in the dataset and exit",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    try:
        # Create converter
        converter = NILMTKConverter(
            input_path=args.input,
            output_dir=args.output,
            max_records=args.max_records,
            target_buildings=args.buildings,
            output_format=args.format,
            target_devices=args.devices,
            strip_nan=args.strip,
        )

        # Show info if requested
        if args.info:
            building_info = converter.get_building_info()
            print("\nDataset Information:")
            print(f"Total buildings: {len(building_info)}")
            for building_num, info in building_info.items():
                print(f"  Building {building_num}: {info['num_meters']} meters")
            return

        # List appliances if requested
        if args.list:
            converter.list_all_appliances(filter=args.devices)
            return

        # Convert dataset
        converter.convert()

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
