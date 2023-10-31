import os
import random

import pandas as pd

from Gravity_Model_FSKx import get_production_potential

##### Definition of input Data #####
## Outbreak sizes ##
list_outbreak_scenario_sizes = [10]
no_of_trials_per_scenario = 1

subfolder = "Outputs"
# As we want to make the artificial Outbreaks reproducible, we set the seed for the generation of random numbers
random.seed(123)


def get_stores(chain_name, all_stores):
    return all_stores[all_stores["chain"] == chain_name]


def get_flow_for_chain(sales_per_cell, selected_stores):
    # First we need to get all cells in which there are two stores:
    total_flow = pd.read_pickle(os.path.join(subfolder, "Flow", "flow.pkl"))

    # select all flows from cells where there is a store of the given chain inside
    selected_flow = total_flow[total_flow.index.isin(selected_stores.index)]

    # These flows are correct unless there is more than the one store of the given chain in any cell
    # First we only selected the cells in which there are more than one store
    multi_store_cells = sales_per_cell[sales_per_cell["stores_count"] > 1]

    selected_flow = selected_flow.merge(
        multi_store_cells["production_potential"],
        left_index=True,
        right_index=True,
        how="left",
    )

    selected_flow = selected_flow.merge(
        selected_stores["sales"], left_index=True, right_index=True, how="left",
    )

    adjusted_rows = (
        selected_flow.loc[selected_flow["production_potential"].notna()]
        .iloc[:, 0:-2]
        .multiply(
            (
                selected_flow.loc[selected_flow["production_potential"].notna(
                )].sales
                / selected_flow.loc[
                    selected_flow["production_potential"].notna()
                ].production_potential
            ),
            axis=0,
        )
    )

    selected_flow = selected_flow[selected_flow["production_potential"].isnull()].iloc[
        :, 0:-2
    ]

    selected_flow = pd.concat([selected_flow, adjusted_rows])

    return selected_flow


def get_cumulative_distribution(flow):
    total_sales = flow.values.sum()

    flow = flow.T

    flow["ingoing_sum"] = flow.sum(axis=1)
    flow["percent"] = flow["ingoing_sum"] / total_sales
    flow["cumulated"] = flow["percent"].cumsum()

    flow = flow.iloc[:, -3:]
    return flow[["ingoing_sum", "percent", "cumulated"]]


def get_location_for_outbreak(cumulative_distribution):
    random_number = random.random()
    return cumulative_distribution[
        cumulative_distribution["cumulated"] > random_number
    ].index[0]


def generate_outbreak(chain_name, no_of_cases, all_stores):
    selected_stores = get_stores(chain_name, all_stores)

    sales_per_cell = get_production_potential(all_stores)

    flow = get_flow_for_chain(sales_per_cell, selected_stores)

    cumulative_distribution = get_cumulative_distribution(flow)

    outbreak_scenario = []

    outbreak_scenario = [
        get_location_for_outbreak(cumulative_distribution) for _ in range(no_of_cases)
    ]

    return outbreak_scenario


def get_xy(outbreak_scenario):
    df = pd.DataFrame({"cell_id": outbreak_scenario})
    population_data = pd.read_pickle(
        os.path.join(subfolder, "Population", "population.pkl")
    )
    df = df.merge(
        population_data[["x_centroid", "y_centroid"]],
        left_on="cell_id",
        right_index=True,
        how="left",
    )

    return df

def main(list_outbreak_scenario_sizes, no_of_trials_per_scenario):
    all_stores = pd.read_pickle(os.path.join(subfolder, "Stores", "stores.pkl"))

    # Number of stores per chain
    chains = all_stores.groupby(["chain"])["chain"].agg("count")

    for chain in chains.index:
        for no_of_outbreak_cases in list_outbreak_scenario_sizes:
            for trial in range(0, no_of_trials_per_scenario):
                outbreak_name = f"{chain}_{no_of_outbreak_cases}_{trial}"

                outbreak_scenario_cells = generate_outbreak(
                    chain, no_of_outbreak_cases, all_stores
                )

                outbreak_scenario = get_xy(outbreak_scenario_cells)

                os.makedirs(os.path.join(subfolder, "Outbreaks"), exist_ok=True)
                outbreak_scenario.to_pickle(
                    os.path.join(
                        subfolder, "Outbreaks", f"Outbreak_{outbreak_name}.pkl"
                    )
                )

if __name__ == "__main__":
    main(list_outbreak_scenario_sizes, no_of_trials_per_scenario)