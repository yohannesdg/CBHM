# Distributed monthly water-balance model: Volta and any African basin

A grid-based (0.05°) monthly hydrologic model of the Volta Basin, with a Python back end
(FastAPI + NumPy/Numba) and a browser front end (Leaflet + Chart.js, served locally).

* **Rainfall:** CHIRPS v2.0 monthly, 0.05°
* **PET:** Modified Hargreaves (Droogers & Allen, 2002) from Tmax/Tmin (TerraClimate by default) and CHIRPS rainfall
* **Soil and groundwater:** abcd model (Thomas, 1981), run in every cell
* **Routing:** D8 flow directions from a priority-flood over the DEM, monthly accumulation to the gauges, optional linear-reservoir lag, open-water evaporation on Lake Volta cells
* **Distributed parameters:** each gauge's own incremental sub-catchment is a parameter zone, calibrated with differential evolution from upstream to downstream
* **Front end:**
  * maps of any flux or state (mean annual, calendar month, a single month, or an anomaly)
  * a rainfall-ribbon time scrubber
  * simulated vs. observed hydrographs, monthly in m³/s or annual volumes in million m³ (MCM), with NSE, KGE, log-NSE, bias and R² for the calibration and validation periods
  * the basin water balance and its closure check
  * a **Water partition** tab: a two-ring pie of where the rainfall goes (evaporation from land and open water, soil storage, quickflow, recharge split into baseflow and groundwater storage) for the whole basin or any sub-basin, and a pie of each sub-basin's share of rainfall, evaporation, streamflow or recharge
  * a **View** menu with three dashboard layouts (Map focus, Balanced, Charts focus) and a Light / Dark / Match system theme; both are remembered in the browser
  * a cell inspector
  * exports to CSV, JSON and ESRI ASCII grid

---

## 1. Install (Windows)

Double-click **`setup_and_run.bat`**. It:

1. looks for a Python 3.10 to 3.13 that **already has** numpy, scipy, FastAPI and uvicorn. It checks `python` on PATH (including the Microsoft Store Python), the `py` launcher and Anaconda. If one is found it is used as-is: nothing is downloaded and conda is never run.
2. If none is ready, it builds `.venv` on top of the best Python found. The venv reuses that Python's packages and downloads only what is missing. pip's output goes to `setup_log.txt`.
3. It remembers the chosen interpreter in `.python_path.txt`, prints a package check (`tools\check_env.py`), and starts the app.

* `setup_and_run.bat prep` also installs xarray, netCDF4 and rasterio, which only `prep\prep_volta.py` needs.
* `setup_and_run.bat rebuild` forgets the choice and searches again.
* After setup, `run_app.bat` alone starts the app.
* numba is optional but makes runs and calibration about 10× faster. Double-click **`install_numba.bat`** (it uses the same Python as the app; about 45 MB, downloaded in resumable parts so a dropped connection doesn't start it over), then restart the app.
* Tests: install `requirements-dev.txt`, then run `python -m pytest`.

**If you prefer a conda environment**, avoid the slow solve by using only conda-forge and the libmamba solver:

```bat
conda env create -f environment.yml --solver=libmamba
```

`environment.yml` already excludes the `defaults` channel. Mixing it with conda-forge is the usual cause of very long solves.
If your conda is older than 23.10, run `conda install -n base conda-libmamba-solver` once.
`run_app.bat` finds an env named `volta-hydro` automatically.

## 2. Try it immediately with the demo basin

```bat
run_app.bat
```

The browser opens at <http://localhost:8000> once the server is ready. With no dataset in `data\`, the page offers **Build demo basin**.
The demo is a *synthetic* basin shaped like the Volta: real-like geography and climate gradients, plus hidden "true"
parameters that generate fake gauge records. It is only for learning the tools. The header shows
"Synthetic demo data" while it is loaded.

## 3. Build the real Volta dataset

**Quick route:** double-click **`download_data.bat`**. It downloads everything below into `data\raw`, then builds `data\volta.npz`. Restart the app afterwards and it opens the real basin. If a download stops, run the file again: finished files are kept, and the DEM and lake downloads resume where they stopped.

| Data | Source | Size |
|---|---|---|
| CHIRPS v2.0 monthly rainfall, 0.05° | IRI Data Library subset of the Volta box | about 60 MB |
| TerraClimate monthly Tmax and Tmin | University of Idaho THREDDS subsets | about 45 MB each |
| HydroSHEDS v1 void-filled DEM, 30″, Africa | hydrosheds.org | 48 MB |
| Natural Earth 1:10m lakes (Lake Volta), public domain | naturalearthdata.com | 2 MB |

The script needs only numpy, scipy and two small readers, `tifffile` and `pyshp` (`requirements-prep.txt`). It uses xarray, netCDF4 and rasterio if they are installed, but doesn't need them.

**The basin is traced from the DEM.** Check the build report: the basin area should be about 400,000 km², and the gauge areas should be close to the published GRDC values (Bamboi 134,200, Nawuni 92,950, Senchi 394,100 km²). If they aren't, pass a basin polygon with `--basin-mask`, for example HydroBASINS exported to GeoJSON.

**Observed discharge** is not public domain for the Volta gauges.
* The open records end before CHIRPS starts in 1981:
  * RivDIS/PANGAEA (CC BY 3.0): Bamboi 1950–1974.
  * Dai & Trenberth (CC BY 4.0): Senchi 1936–1984, regulated by Akosombo from 1964.
* For calibration, request the stations from the GRDC Data Portal (free for research; the link arrives by email): Bamboi 1531100, Nawuni 1531450, Sabari 1531800 and Senchi 1531700. You can also ask the Ghana Hydrological Authority or the Volta Basin Authority.
* **Open records (no account needed):** **Data tab → Find open gauge records** uses ADHI, the African Database of Hydrometric Indices (Tramblay et al., 2021, CC BY 4.0, https://doi.org/10.23708/LXGXQ9).
  * It has monthly mean discharge for 1,466 African stations.
  * For the Volta it covers Nawuni (1981–2007, 271 months), Saboba (249 months), the Black Volta above Bui dam (204 months, close to Bamboi), Dapola, Porga, Pwalagu and others. Records mostly end around 2007; Senchi and Bamboi themselves stop in the 1970s.
  * The app downloads about 7 MB once, into `data\raw\adhi`. It snaps each station in the open basin to the model's river network, and accepts it only if the catchment areas agree within 25 %.
  * For each station you choose: add it as a new gauge, use it as records for an existing gauge, or skip it. A station is offered for an existing gauge when it has the same name (catchment within 15 %) or lies within a few cells with a catchment within 10 %.
  * Selected by default: stations with at least 7 years of records in the model period and a catchment of at least 2 % of the basin.
  * Not selected: reservoir records (volumes in Mm³, "barrage"), the second copy of a station listed twice (SIEREM and GRDC), and stations on the same river as a chosen gauge with nearly the same catchment (within 15 %), since each gauge becomes a calibration zone. Every skipped station shows the reason, and you can still pick it.
  * For the Volta, the defaults keep about a dozen gauges. Bui, Nawuni and Saboba become records for the configured Bamboi, Nawuni and Saboba gauges.
  * Adding stations rewrites the dataset with the new gauges and records. Because gauges define the calibration zones, any per-sub-basin values are reset; whole-basin values are kept.
  * Calibration and validation periods then follow the records: the years where at least a quarter of the measured gauges report are split about 60/40, and the Calibrate tab shows the result. Calibration stops with a clear message if no gauge has 24 months of records in the calibration period.
* Then add it in the app: **Data tab → Add observed discharge**. It accepts the CSV below or a GRDC export file as downloaded (one station per file; pick the gauge from the list if the station name doesn't match). The records are saved into the dataset and the model reruns, so fit scores and calibration become available. You can also rebuild with `--qobs your_file.csv`.

**Discharge CSV:** a `date` column (`YYYY-MM` or `YYYY-MM-DD`) and one column per gauge, named as in the config (Bamboi, Nawuni, Saboba, Senchi…), in m³/s. Daily data are averaged to months, and a month is kept only if at least 80 % of its days are present.

```csv
date,Bamboi,Nawuni,Saboba,Senchi (below Akosombo)
1983-01-01,45.2,12.1,3.4,
...
```

Manual commands, with other options:

```bat
python prep\prep_volta.py download               rem resumable
python prep\prep_volta.py build --qobs D:\data\volta_discharge.csv
python prep\prep_volta.py build --basin-mask D:\gis\volta.geojson --res 0.1 --start 1991-01 --end 2020-12
python prep\prep_volta.py check
```

## 3b. Any river basin in Africa (HydroBASINS levels 2–5)

Open the **Basins** tab in the app.

1. **First time only:** click **Download basin outlines**. This fetches HydroSHEDS HydroBASINS Africa (`hybas_af_lev01-12_v1c.zip`). The download resumes if it stops. The app keeps levels 2–5 and makes light outlines for the map (`data\basins`).
2. **Pick the level** with the slider. Level 2 gives whole river systems; level 5 gives tributary catchments.
3. **Click a basin.** With *Include everything upstream* ticked, the catchment is that unit plus every unit draining into it (HydroBASINS `NEXT_DOWN`). The panel shows the area, the model grid and the download size.
4. **Prepare and open.** The app:
   * downloads CHIRPS and TerraClimate for that basin's box only;
   * builds the model, using the Africa DEM and Natural Earth lakes from `download_data.bat`;
   * zooms to the basin and opens its dashboard.

   Prepared basins are listed in the tab and open instantly next time.

How a basin is set up:
* **Grid size:** the model resolution is chosen so the basin has at most about 20,000 cells: 0.05° for basins up to about 600,000 km², then 0.1°, 0.25° or 0.5°. For grids coarser than 0.05°, the IRI Data Library regrids CHIRPS on the server, so even the Nile or Congo download only tens of MB.
* **Parameter zones** are the HydroBASINS units inside the catchment. A single unit is split into its next-level units. The Sub-basins tab lists them by Pfafstetter code.
* **Outlet:** found automatically, as the edge cell that drains the most area. Inland-draining basins are routed to their lowest edge point.
* **No gauge data:** these basins have no observed discharge, so the hydrograph shows simulated outflow and calibration is disabled until discharge records are added.

Command line, for the same steps:

```bat
python prep\basin_data.py catalog
python prep\basin_data.py plan --level 3 --id 1030022420
python prep\basin_data.py prepare --level 3 --id 1030022420
```

**Where basin rainfall comes from.** CHIRPS is first requested from the IRI Data Library in chunks of up to three years, each answered in seconds. IRI has been throttling heavy traffic since it started requiring sign-in for its web pages; data downloads themselves still need no account. If a chunk gets no answer within 45 seconds, preparation switches to the Climate Hazards Center's cloud-optimized GeoTIFFs (`data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/cogs`). No sign-in is needed there: the app reads only the 512-pixel tiles that cover the basin, about 0.5–1 MB per tile per month, using several parallel connections. Tiles are kept in `data\raw\chirps_cog`, so neighbouring basins and later runs reuse them. The plan shows that fallback size before you start, and **Details** under the progress bar lists every step.

## 3c. Climate scenarios

The **Scenarios** tab reruns the current model with its current parameters under a simple change in climate, then compares the result with the current run:

* **Rainfall change** (−30 to +30 %): every month's rainfall is scaled.
* **Temperature change** (0 to +4 °C): Tmax and Tmin are raised, and Modified Hargreaves PET is recomputed. Its rain term follows the new rainfall too.
* **Extra PET change** (optional): scales PET on top of the warming.
* **Presets:** drier and hotter (−10 %, +2 °C), wetter and hotter (+10 %, +2 °C), hotter only (+2 °C), and severe (−20 %, +3 °C).

The output covers:

* basin rainfall, PET, actual evaporation and runoff per year;
* outlet flow in m³/s and million m³ per year;
* the wettest month and the dry season (the 3 lowest months);
* the runoff ratio;
* the mean flow by calendar month and the yearly outflow volume, now and under the scenario;
* the change at each gauge;
* the rainfall sensitivity of flow, when only rainfall changes: % change in flow per 1 % change in rainfall.

**Monthly flows (CSV)** exports the outlet and gauge series for both runs. On the demo Volta, 10 % less rain lowers outlet flow by about 35 %, and adding 2 °C of warming makes it about 40 %. Semi-arid basins amplify rainfall changes like this.

These are "delta change" scenarios: the observed 1981–2023 climate shifted uniformly. Changes in rainfall timing or in the number of dry spells are not represented. For projections, the deltas can be taken from climate model ensembles (for example CMIP6 changes for West Africa).

**Save as run** keeps the scenario as a saved run (section 3d), so it can be compared with any other run on the hydrograph.

## 3d. Using the interface

**Workflow steps (left).** The left tabs are numbered in the order the work is done: **1 Basin**, **2 Gauges** (observed records), **3 Model** (model choice and parameters), **4 Calibrate**. Each step shows a short status, such as "12/20 with data", "abcd, edited" or "calibrated 14:20", and a green tick once it is done.

**Key numbers (top right).** Rainfall, PET, actual ET, outflow (m³/s; hover for mm/yr and million m³/yr) and runoff ratio, with a bar showing where the rain goes. "✓ balance closes" confirms the water balance closes; hover to see the closure error.

**Fit strip.** One chip per gauge with records shows its KGE, coloured and worded: good (≥ 0.7), fair (≥ 0.5) or poor. Click a chip to open that gauge's hydrograph. "+N without records" opens step 2.

**Results groups.** The results are grouped:
* **Flows:** Hydrograph, Seasonal cycle.
* **Water balance:** Partition, Sub-basins.
* **Scenarios.**
* **Cell.**

The views inside a group are switched on the right of the tab bar, and each group remembers the last view used.

**Saved runs.** In step 3, give the current parameters a name and press **Save run**. Runs are stored per dataset in `data\runs\<dataset>.json`. Each run lists its model, runoff ratio and median KGE, with four actions:
* **Compare:** draws the run as a dashed line on the hydrograph, with a score table.
* **Use:** makes its parameters, and its model, the current ones.
* **Rename.**
* **Delete:** asks for a second click.

The hydrograph's **Compare with** list offers the other model (with its last parameters) and every saved run, including saved climate scenarios. A saved run's flows are recomputed when it is compared, so its scores always use the current observed records.

## 4. Model description

For each cell and month (P and PET in mm):

**Modified Hargreaves PET**

ETo = 0.0013 · 0.408 · Ra · (Tavg + 17.0) · (Tmax − Tmin − 0.0123 P)^0.76  [mm day⁻¹], times days in month

* Ra is FAO-56 extraterrestrial radiation for the cell latitude on the 15th of the month.
* The bracket is clipped at zero in very wet months.
* PET = kc · ETo.

**abcd soil and groundwater**

| Step | Equation |
|---|---|
| Available water | W = P + S₍ₜ₋₁₎ |
| ET opportunity | Y = (W+b)/2a − √[((W+b)/2a)² − W·b/a] |
| Soil moisture | S = Y·e^(−PET/b) |
| Actual ET | ET = Y − S |
| Recharge | R = c·(W − Y) |
| Direct runoff | Qd = (1−c)·(W − Y) |
| Groundwater | G = (G₍ₜ₋₁₎ + R)/(1+d) |
| Baseflow | Qb = d·G |

* Initial S and G come from cycling the mean annual cycle of the first five years three times.
* The balance P = ET + Qd + Qb + ΔS + ΔG holds exactly. The app shows the basin closure error, which should be about 0 mm.

| Parameter | Meaning | Range |
|---|---|---|
| a | runoff before soil saturation; very sensitive close to 1 in this climate | 0.85–1.0 |
| b | soil water + ET capacity, mm | 100–3000 |
| c | share of surplus recharging groundwater | 0–1 |
| d | groundwater recession, 1/month | 0.005–1 |
| kc | multiplier on Modified Hargreaves ETo | 0.6–1.3 |
| kw | open-water evaporation coefficient for lake cells (basin-wide) | 0.7–1.3 |
| k_route | linear-reservoir lag at the gauges, months, 0 = off (basin-wide) | 0–6 |

**Routing.**
* Runoff is converted to m³/s and accumulated down the D8 network in topological order.
* All runoff is assumed to reach the gauge within the month.
* Lake cells add P − kw·ETo to the flow passing through them. Flow can't go below zero, and evaporation that can't be supplied is reported.
* The optional linear reservoir represents channel and floodplain storage (the White Volta floodplains, for example).

**Calibration.**
* Gauges are processed upstream first. Each gauge's incremental zone gets its own a, b, c, d (and kc if you choose).
* The upstream gauges' simulated flows are injected as inflow, so each trial only runs the zone's own cells. This is exact, and it is checked in the tests.
* Objectives: KGE, NSE, log-NSE, or ½ KGE + ½ log-NSE. Calibration and validation periods default to 1983–2005 and 2006–2023 (`config\volta.json`).

**Choosing the water-balance model.** The Parameters tab has a model picker:

* **abcd** (default): the model described above.
* **NAM (monthly):** the DHI NAM structure adapted from `models/NAM_Model-master` to a monthly step, run in every cell. It has a surface store (Umax), a root zone (Lmax), overland flow (CQOF, TOF), interflow (CKIF, TIF), recharge (TG) and a groundwater store draining as baseflow (CKBF), plus the same PET multiplier kc.
  * Time constants are in months.
  * CK12 (routing of the fast flows) is dropped, because those flows reach the river within the month.
  * Snow is left out. In the original code it also double-counted rain between 0 °C and the snow threshold.
  * The water balance closes exactly in every cell and month (see `hydro/nam.py`).

Each model keeps its own parameter set, so switching back and forth doesn't lose your values. Calibration, maps, the water partition and the exports all work with either model.

**Compare with…** on the Hydrograph tab runs the other model with its own last parameters. It draws that model as a dashed line and adds a table of KGE, NSE, bias and mean flow for both models. To add another model, give it a parameter list and a run function in `hydro/models.py`.

NumPy runs NAM on the 13,000-cell Volta in about 2 seconds. With numba (`python -m pip install numba`) it takes well under a second, and calibration is about 10 times faster.

**Known limitations**
* Simulated flows are **naturalized**. Akosombo, Kpong, Bui, Bagré, Kompienga and small reservoirs are not operated in the model, so compare Senchi with naturalized or pre-1964 flows, or judge it with care.
* Monthly steps with no explicit wetland storage. Use `k_route` or calibrate per zone to compensate.
* CHIRPS and TerraClimate have their own biases. kc can absorb a systematic PET bias.
* One parameter set per zone. Finer regionalization (by soil or land cover) would need parameter grids, which the model already accepts internally as per-cell arrays.

## 5. Project layout

```
hydro/      model core: grid, pet (Modified Hargreaves), abcd and nam (monthly NAM), models (model registry), routing (priority-flood D8,
            accumulation, lakes), model (run, balance), calibrate (zone-wise differential evolution),
            dataset (npz I/O, basin builder), synthetic (demo basin), metrics,
            geo (shapefiles, rasterising), basins (HydroBASINS catalogue, upstream tracing)
backend/    FastAPI app (REST API + serves the front end)
frontend/   index.html, app.js, styles.css, vendor/ (Leaflet 1.9.4, Chart.js 4.4.1, Archivo font; works offline)
prep/       prep_volta.py: CHIRPS/TerraClimate/DEM/lakes download, regridding, basin, gauges -> data/volta.npz
            basin_data.py: HydroBASINS catalogue and on-demand datasets for any African basin
config/     volta.json: bbox, outlet, gauges, periods
tests/      pytest suite (PET, abcd balance, routing, calibration, API, prep end-to-end with fixtures)
```

The map basemap tiles (CARTO/OpenStreetMap) need internet. Everything else runs offline.

## 6. API (all JSON unless noted)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | loaded dataset, available datasets, numba |
| POST | `/api/datasets/synthetic` `{res}` | build the demo basin |
| POST | `/api/datasets/load` `{name}` · `/api/datasets/upload` (file) | switch dataset |
| GET | `/api/meta` | grid, dates, gauges, zones, parameter ranges, periods |
| POST | `/api/run` `{params}` | run; returns balance, basin series, gauge series and metrics (`params.model` = `abcd` or `nam`) |
| POST | `/api/model` `{model}` | switch the water-balance model; each model keeps its last parameter set |
| GET | `/api/compare?model=` | run the other model with its last parameters, for comparison (the current results are not changed) |
| GET | `/api/map?var=&mode=annual\|clim\|month\|anomaly&t=&month=` | map values |
| GET | `/api/static/{dem\|acc\|zone\|lake}` | basin layers |
| GET | `/api/cell?row=&col=` | one cell's series and parameters |
| POST | `/api/calibrate` `{names, objective, cal_start, cal_end, maxiter, popsize, gauges}` | start calibration in the background |
| GET | `/api/calibrate/status` · POST `/api/calibrate/cancel` · POST `/api/calibrate/apply` | follow, stop, or apply a calibration |
| POST | `/api/observed?name=&gauge=` (raw file body) | add observed discharge (CSV or GRDC export) to the gauges and save it in the dataset |
| GET | `/api/adhi/candidates` · POST `/api/adhi/fetch` · POST `/api/adhi/add` `{choices}` | open ADHI gauge records: stations in the basin, download, add or merge |
| POST | `/api/scenario` `{rain_pct, temp_c, pet_pct}` · GET `/api/export/scenario.csv` | climate scenario compared with the current run; its monthly flows |
| GET · POST | `/api/runs` `{label, climate?}` | list saved runs; save the current parameters (optionally with a climate change) |
| GET · PATCH · DELETE | `/api/runs/{id}` | a saved run's flows and scores, recomputed (same shape as `/api/compare`); rename; delete |
| POST | `/api/runs/{id}/apply` | make a saved run's model and parameters current |
| GET | `/api/export/gauges.csv` · `basin.csv` · `annual.csv` · `params.json` · `map.asc` | downloads; `annual.csv` gives calendar-year volumes in million m³ |

Parameters JSON:

```json
{"global": {"a": 0.995, "b": 700, "c": 0.4, "d": 0.1, "kc": 1.0},
 "zones": {"1": {"b": 820, "d": 0.05}},
 "kw": 1.05, "k_route": 0.0}
```

## 7. Tests

```bat
python -m pytest
```

## References

* Droogers, P. & Allen, R.G. (2002). Estimating reference evapotranspiration under inaccurate data conditions. *Irrigation and Drainage Systems* 16, 33–45.
* Thomas, H.A. (1981). Improved methods for national water assessment. USGS. Alley, W.M. (1984). *Water Resources Research* 20(8).
* Funk, C. et al. (2015). The climate hazards infrared precipitation with stations (CHIRPS). *Scientific Data* 2, 150066.
* Abatzoglou, J.T. et al. (2018). TerraClimate. *Scientific Data* 5, 170191.
* Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper 56.
* Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood. *Computers & Geosciences* 62, 117–127.
* Tramblay, Y. et al. (2021). ADHI: the African Database of Hydrometric Indices (1950–2018). *Earth System Science Data* 13, 1547–1560.
* Gupta, H.V. et al. (2009). Decomposition of the MSE and NSE (KGE). *Journal of Hydrology* 377, 80–91.
