import math
import os

import numpy as np
import pandas as pd
from ipfn import ipfn
from scipy.spatial import distance_matrix

##### Definition of input Data #####
## Shops Data ##
# only one chain with 5 stores distributed randomly
# The coordinates go from 0 to 1000
# The shops shouldn't be on a round number cause otherwise they're within 4 cells at the same time
# (all lists need to be the same length)
x_coord = [0.112, 0.823, 0.888, 0.105, 0.487]
y_coord = [0.198, 0.112, 0.846, 0.855, 0.537]
Chain = ["Chain 1", "Chain 1", "Chain 1", "Chain 1", "Chain 1"]
Sales = [1000, 1000, 1000, 1000, 1000]

## Population Data ##
# uniform population of 5 in each cell (500 total)
population_per_cell = 5

## Data on Shopping behavior ##
# shopping distance: 0.4 km
empirical_mean_shopping_distance = 0.4  # all units are in km
tolerance = 0.001

subfolder = "Outputs"

def check_input_data(x_coord, y_coord, Chain, Sales):
    # check whether all lists have the same length
    lists = [x_coord, y_coord, Chain, Sales]
    if not all(len(l) == len(lists[0]) for l in lists):
        raise ValueError(
            "Not all lists that define the shops data have the same length"
        )

def sanity_check(data, description):
    if np.isnan(data).any():
        raise ValueError(f"NaN values found in {description}.")
    if np.isinf(data).any():
        raise ValueError(f"Infinity values found in {description}.")
    if (data < 0).any():
        raise ValueError(f"Negative values found in {description}.")

def create_population_df(population_per_cell):
    population_data = {
    "x_centroid": [i * 0.1 + 0.05 for i in range(10) for _ in range(10)],
    "y_centroid": [i * 0.1 + 0.05 for _ in range(10) for i in range(10)],
    "population": population_per_cell * 100,
    "cell_id": [float(i + 1) for i in range(100)],
    }
    
    df_population = pd.DataFrame(population_data)
    df_population['cell_id'] = df_population['cell_id'].astype(int)
    
    if df_population.index.name != "cell_id":
        df_population.set_index("cell_id", inplace=True)
    
    # save to pkl
    output_dir = os.path.join(subfolder, "Population")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_population.to_pickle(os.path.join(output_dir, "population.pkl"))
    
    return df_population

def create_shops_df(df_population, x_coord, y_coord, Chain, Sales):
    df_shops = pd.DataFrame(
        {
            "x": x_coord,
            "y": y_coord,
            "chain": Chain,
            "sales": Sales,
            "cell_id": "",
        }
    )
    for ind in df_shops.index:
        df_shops.loc[ind, "cell_id"] = (
            df_population[
                ((df_population["x_centroid"] - 0.05) <= df_shops.x[ind])
                & ((df_population["x_centroid"] + 0.05) >= df_shops.x[ind])
                & ((df_population["y_centroid"] - 0.05) <= df_shops.y[ind])
                & ((df_population["y_centroid"] + 0.05) >= df_shops.y[ind])
            ].index.values
        )[0]
        
    if df_shops.index.name != "cell_id":
        df_shops.set_index("cell_id", inplace=True)
        
    # save to pkl
    output_dir = os.path.join(subfolder, "Stores")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_shops.to_pickle(os.path.join(output_dir, "stores.pkl"))

    return df_shops


def get_distance_matrix(production, consumption):
    production_centroids = pd.concat(
        [production.x_centroid, production.y_centroid], axis=1
    )
    consumption_centroids = pd.concat(
        [consumption.x_centroid, consumption.y_centroid], axis=1
    )

    arr_distance = distance_matrix(
        production_centroids,
        consumption_centroids,
    )
    # in-cell distance shouldn't be zero and is set according to Czuber (1884)
    arr_distance[arr_distance == 0] = (128 / (45 * math.pi)) * 0.05

    sanity_check(arr_distance, "distance matrix")

    return arr_distance


def get_production_potential(shops_data):
    production_potential = shops_data.groupby("cell_id").agg(
        stores_count=("sales", "count"), production_potential=("sales", "sum"),
    )
    sanity_check(production_potential.production_potential, "production potential")
    return production_potential


def get_consumption_potential(population_data, total_revenue):
    total_population = population_data["population"].sum()
    if total_population == 0:
        raise ValueError("Total population cannot be zero.")
    consumption_potential = population_data.copy()
    consumption_potential["consumption_potential"] = (
        consumption_potential["population"].divide(total_population)
    ).multiply(total_revenue)
    consumption_potential = consumption_potential[
        consumption_potential["population"] != 0
    ]
    sanity_check(consumption_potential.consumption_potential, "consumption potential")
    return consumption_potential


def furness_model(
    beta: float, dist_matrix, production_potential, consumption_potential
):
    dM = np.exp(-beta * dist_matrix)

    prod_pot_new = production_potential.production_potential.to_numpy()
    cons_pot_net = consumption_potential.consumption_potential.to_numpy()

    aggregates = [
        prod_pot_new,
        cons_pot_net,
    ]
    dimensions = [[0], [1]]
    IPF = ipfn.ipfn(dM, aggregates, dimensions)

    dM = IPF.iteration()
    flowMatrix = dM
    return flowMatrix


def get_weighted_dist(flow_matrix, dist_matrix):
    WeightDist = np.sum(flow_matrix * dist_matrix) / (np.sum(flow_matrix))
    return WeightDist


def add_indices(flow, production_potential, consumption_potential):
    df_flow = pd.DataFrame(
        flow,
        columns=consumption_potential.index,
        index=production_potential.index,
    )
    return df_flow


def hyman_model(
    empirical_mean_shopping_distance, tolerance, population_data, shops_data
):
    """calibrates the parameter (beta) of a gravity model. This parameter is the input for the furness-algorithm to calculate the flow of goods.
        The exponential distance model is hardcoded
    Args:
        empirical_mean_shopping_distance (float): used to compare the modeled mean distance
        tolerance (float): needed to decide when a satisfactory solution is reached

    Returns:
        flow(numpy.ndarray): _description_
    """
    beta_list = []  # keeping track of the betas
    modeled_means_list = []  # keeping track of the average of the modeled flow distance
    count_loops = 0

    # initializing Hyman with beta_0
    beta_0 = 1.0 / empirical_mean_shopping_distance
    beta_list.append(beta_0)

    production_potential = get_production_potential(shops_data)  # rows
    total_revenue = production_potential["production_potential"].sum()
    consumption_potential = get_consumption_potential(population_data, total_revenue)
    production_potential = production_potential.merge(
        population_data.drop(columns=["population"]), on="cell_id", how="left"
    )

    dist_matrix = get_distance_matrix(production_potential, consumption_potential)

    flow_0 = furness_model(
        beta_0, dist_matrix, production_potential, consumption_potential
    )

    modeled_mean_shopping_distance = get_weighted_dist(flow_0, dist_matrix)
    modeled_means_list.append(modeled_mean_shopping_distance)

    if (
        abs(empirical_mean_shopping_distance - modeled_means_list[count_loops])
        <= tolerance
    ):
        flow = flow_0
        modeled_mean_current = get_weighted_dist(flow, dist_matrix)
        modeled_means_list.append(modeled_mean_current)
        
    while (
        abs(empirical_mean_shopping_distance - modeled_means_list[count_loops])
        > tolerance
    ):
        if count_loops == 0:
            beta_1 = (
                beta_0
                * modeled_means_list[count_loops]
                / empirical_mean_shopping_distance
            )
            beta_list.append(beta_1)
        elif count_loops > 0:
            beta_next = np.abs(
                (
                    (
                        (
                            empirical_mean_shopping_distance
                            - modeled_means_list[count_loops - 1]
                        )
                        * beta_list[count_loops]
                        - (
                            empirical_mean_shopping_distance
                            - modeled_means_list[count_loops]
                        )
                        * beta_list[count_loops - 1]
                    )
                    / (
                        modeled_means_list[count_loops]
                        - modeled_means_list[count_loops - 1]
                    )
                )
            )
            beta_list.append(beta_next)
        beta_current = beta_list[count_loops + 1]

        flow = furness_model(
            beta_current, dist_matrix, production_potential, consumption_potential
        )
        modeled_mean_current = get_weighted_dist(flow, dist_matrix)
        modeled_means_list.append(modeled_mean_current)

        count_loops += 1

        # break if in local minimum and check if any dist was closer to the empirical mean shopping distance
        if count_loops > 20:
            if (
                abs(
                    modeled_means_list[count_loops]
                    - modeled_means_list[count_loops - 5]
                )
            ) < 0.001:
                beta_best = beta_list[modeled_means_list.index(min(modeled_means_list))]
                flow = furness_model(
                    beta_best, dist_matrix, production_potential, consumption_potential
                )
                break

        # break if minimization routine explodes due to numerical issues
        if beta_current > 50:
            beta_best = beta_list[modeled_means_list.index(min(modeled_means_list))]
            flow = furness_model(
                beta_best, dist_matrix, production_potential, consumption_potential
            )
            break
        print(
            "On the %sd. iteration: distance between the modeled and the empirical mean shopping distance is down to %3.4f"
            % (
                count_loops,
                abs(empirical_mean_shopping_distance - modeled_means_list[count_loops]),
            )
        )

        if np.isnan(empirical_mean_shopping_distance):
            raise Exception(
                "Something went wrong, the given empirical mean shopping distance returned nan!"
            )
        if np.isnan(modeled_means_list[count_loops]):
            raise Exception(
                "Something went wrong, the current modeled mean shopping distance is nan!"
            )

    beta_best = beta_list.pop()

    # Sanity Check
    tol_this_time = np.abs(empirical_mean_shopping_distance - modeled_mean_current)
    tol_best = np.abs(
        [empirical_mean_shopping_distance - d for d in modeled_means_list]
    ).tolist()
    if tol_this_time > tol_best[tol_best.index(min(tol_best))]:
        beta_best = beta_list[tol_best.index(min(tol_best))]
        flow = furness_model(
            beta_best, dist_matrix, production_potential, consumption_potential
        )
    print(
        "On the last iteration (%2d.): tolerance is down to %3.4f"
        % (tol_best.index(min(tol_best)), tol_best[tol_best.index(min(tol_best))])
    )
    print("Beta is " + str(beta_best))

    sanity_check(flow, "flow matrix")
    if not np.allclose(
        flow.sum(axis=1), production_potential.production_potential.to_numpy(), atol=2,
    ):
        raise ValueError("Row sums do not match production potential.")
    if not np.allclose(
        flow.sum(axis=0),
        consumption_potential.consumption_potential.to_numpy(),
        atol=2,
    ):
        raise ValueError("Column sums do not match consumption potential.")
    flow_end = add_indices(flow, production_potential, consumption_potential)

    return flow_end


def main(x_coord, y_coord, Chain, Sales):
    check_input_data(x_coord, y_coord, Chain, Sales)
    df_population = create_population_df(population_per_cell)
    df_shops = create_shops_df(df_population, x_coord, y_coord, Chain, Sales)
    flow = hyman_model(
        empirical_mean_shopping_distance, tolerance, df_population, df_shops
    )

    os.makedirs(os.path.join(subfolder, "Flow"), exist_ok=True)
    flow.to_pickle(os.path.join(subfolder, "Flow", "flow.pkl"))


if __name__ == "__main__":
    main(x_coord, y_coord, Chain, Sales)
