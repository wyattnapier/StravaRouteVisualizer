# Strava Route Visualization Project

### strava_auth.py

use this file to get the strava refresh token

run it with `python strava_auth.py --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET` in the terminal

### strava_to_3d.py

use this file to do the majority of the pipeline

beforehand make sure to install the necessary dependencies: `pip install requests numpy scipy rasterio`

then it is also important to make sure you add the relevant variables to your terminal's set of local variables as follows

```
export STRAVA_CLIENT_ID=12345
export STRAVA_CLIENT_SECRET=abc...
export STRAVA_REFRESH_TOKEN=def...
export OPENTOPO_API_KEY=ghi...

```

then you can actually run the application with `python strava_to_3d.py --activity-id YOUR_ACTIVITY_ID`

the activity id flag is optional -- otherwise it defaults to your most recent activity
