// ==UserScript==
// @name         Download Dota Hero Images
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       Mohsen Heydari
// @match        https://website
// @grant        GM_download
// ==/UserScript==

(function() {
    'use strict';
    let links = Array.prototype.slice.apply(document.querySelectorAll("#mw-content-text td a"));
    links.map( el => {

        GM_download({
            url: el.childNodes[0].src,
            name: `dota2/${el.title}.png`
        });

        return el;
    } );
})();