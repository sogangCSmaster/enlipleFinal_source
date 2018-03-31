module.exports = {
  /**
   * Application configuration section
   * http://pm2.keymetrics.io/docs/usage/application-declaration/
   */
  apps : [

    // First application
    {
      name      : 'batch',
      script    : 'batch.py',
      interpreter : 'python3'
    }
  ]
};
