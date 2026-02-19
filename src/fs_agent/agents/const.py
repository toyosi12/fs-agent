spec = """
metadata:
  name: Acme Tasks
  summary: Simple full-stack TODO manager with sharing features.
  owner: platform@acme.dev
  version: 0.1.0
backend:
  language: javascript
  framework: express
  style: rest
  endpoints:
    - name: List tasks
      method: GET
      path: /api/tasks
      description: Return tasks filtered by owner
      response_schema:
        tasks: Task[]
    - name: Create task
      method: POST
      path: /api/tasks
      description: Create a task for the current user
      request_schema:
        title: string
        dueDate: string
      response_schema:
        task: Task
      errors:
        - 400 invalid payload
        - 401 unauthorized
  data_models:
    - name: Task
      description: Individual task entity
      fields:
        id: string
        title: string
        completed: boolean
        dueDate: string
frontend:
  language: javascript
  framework: react
  styling: tailwind
  routes:
    - path: /
      description: Dashboard showing task list
      consumes:
        - GET /api/tasks
      components:
        - TaskList
        - TaskSummary
    - path: /new
      description: Form to create tasks
      consumes:
        - POST /api/tasks
      components:
        - TaskForm
infra:
  ci: github-actions
  cd: fly-io
  targets:
    - name: dev
      environment: dev
      description: Development preview app
      runtime: docker
    - name: prod
      environment: prod
      description: Production deployment
      runtime: docker
"""