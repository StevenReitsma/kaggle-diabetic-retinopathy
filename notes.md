####Redis
* Starting Redis
`sudo service redis_6379 start`
* CLI
`redis-cli`

####Kill all python (also removes reserved GPU memory)
When `killall python` doesn't cut it:
`ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9`
