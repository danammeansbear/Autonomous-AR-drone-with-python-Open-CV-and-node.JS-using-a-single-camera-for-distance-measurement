var arDrone = require('ar-drone');
var client  = arDrone.createClient();

client.takeoff();

client
  .after(50, function() {
    this.right(0.5);
  })
  .after(3000, function() {
    this.stop();
   
  });