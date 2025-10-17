#!/bin/bash

# to use type into terminal
# chmod +x restart_service.sh
# ./restart_service.sh

# Define the services to restart
services=(
  "feed-client"
  "binary_pricer_daemon"
  "binary_pricer_queue"
  "binary_rpc_redis_general"
  "binary_riskd"
  "binary_expiryd"
  "binary_starman_bom-backoffice"
  "binary_websocket_api"
)

# Iterate over the services and restart them
for service in "${services[@]}"
do
  sudo service "$service" restart &
done

# Restart the rpc_redis service
qa_restart_service rpc_redis

# Restart the pricer service
qa_restart_service pricer

# Wait for all services to complete
wait

# All services have been restarted
echo "All services restarted successfully."
