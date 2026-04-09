# Flask APIs and Streamlit Apps

This project serves the model workflows through Flask APIs and Streamlit dashboards. The APIs handle backend prediction or recommendation logic. The Streamlit apps provide browser-based interfaces for users.

## Important Folders

| Folder | Purpose |
| --- | --- |
| `MLops pipeline/flask_apps` | Flask API applications |
| `MLops pipeline/streamlit` | Streamlit dashboard applications |
| `MLops pipeline/gateway` | Nginx local gateway routing |

## Flask APIs

| API | Port | Main purpose |
| --- | --- | --- |
| Flight API | `5002` | Flight price prediction and route summaries |
| Hotel API | `5001` | Popular hotels, similar hotels, and user recommendations |
| Gender API | `5003` | Gender classification by name |

## Streamlit Apps

| App | Port | Purpose |
| --- | --- | --- |
| Flight Streamlit | `8501` | Flight price prediction dashboard |
| Hotel Streamlit | `8502` | Hotel recommendation dashboard |
| Gender Streamlit | `8503` | Gender classification dashboard |

## Local Gateway

The gateway gives one local entry point for the browser:

```text
http://localhost:8090
```

The gateway routes app and API paths to the right service.

## Run Main Apps

From the `MLops pipeline` folder:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8090
```

## Direct Service URLs

| Service | URL |
| --- | --- |
| Flight API | `http://localhost:5002` |
| Flight Streamlit | `http://localhost:8501/app/flight` |
| Hotel API | `http://localhost:5001` |
| Hotel Streamlit | `http://localhost:8502/app/hotel` |
| Gender API | `http://localhost:5003` |
| Gender Streamlit | `http://localhost:8503/app/gender` |

## Expected Output

After startup, the APIs should respond to health checks and the Streamlit apps should open in the browser. The dashboards allow users to interact with the saved model workflows without running development workflow cells manually.

## Notes

The API and UI services are separate on purpose. This keeps backend logic and browser interface logic easier to manage, test, and deploy.
