import socketio
import eventlet
import datetime
import logging
eventlet.monkey_patch()
from flask import Flask, render_template
import sys
import batch

sys.path.append('./config')
from pyredis import Redis
from db import DB
from config import load_config

try:
    db = DB()
    redis = Redis()
    redis_config = load_config()['redis']
    if redis_config['password']:
        manager = socketio.RedisManager('redis://:' + redis_config['password'] + '@' + redis_config['host'] + ':' + str(redis_config['port']))
    else:
        manager = socketio.RedisManager('redis://' + redis_config['host'] + ':' + str(redis_config['port']))
    queue = []
    sio = socketio.Server(client_manager=manager)
    app = Flask('socketio')
except Exception as e:
    raise e


@sio.on('connect', namespace='/enliple')
def connect(sid, environ):
    logging.debug('connect')

@sio.on('disconnect', namespace='/enliple')
def disconnect(sid):
    logging.debug('disconnect', sid)

@app.route('/')
def index():
    return 'port open'

def web_start(state, port):
    global app
    global sio
    # socketio server start

    # start batch_single

    sapp = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(('', port)), sapp)
    except Exception as e:
        print(e)

        state.value = 1
