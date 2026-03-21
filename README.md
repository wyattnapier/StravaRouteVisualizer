# Strava Route Visualization Project

### Relevant files and folders

- `strava_auth.py`: use this file to get the strava refresh token (only need to do once at the start)
- `strava_to_3d.py`: use this file to run the large majority of the pipeline
- `/stl_output`: contains all output files but the 3 most relevant ones are the `.stl` files which when combined create the final product
  - other data that is stored in this file is used for testing and could also be used as cached data to accelerate future processing

### Steps to run it

1) install the necessary dependencies: `pip install requests numpy scipy rasterio python-dotenv`
2) ensure that `.env` file is populated with api key for openTopo, the client ID and the client secret
3) run `python strava_auth.py` which will automatically add the strava refresh token to the `.env` file
4) create the `.stl` files for baseplate, route, and terrain by running `python strava_to_3d.py --activity-id YOUR_ACTIVITY_ID`
  - the activity id flag is optional. without providing it, the program defaults to your most recent activity