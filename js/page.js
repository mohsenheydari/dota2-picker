let head = document.querySelector("head");
let script = document.createElement("script");
script.type = "text/javascript";

function bindEvents(backend) {
  reset = document.querySelector('button[type="reset"]');
  reset.addEventListener("click", function() {
    backend.reset_clicked_slot();
  });
}

script.onload = function() {
  new QWebChannel(qt.webChannelTransport, function(channel) {
    let backend = channel.objects.backend;
    bindEvents(backend);
  });
};

script.src = "qrc:///qtwebchannel/qwebchannel.js";
head.appendChild(script);
