// Create version selector for documentation top bar.
(function() 
{
 var url = window.location.href;

  // If doc is run localy

  var root_url = url.substring(0, url.search('/' + yann_dir)) + '/';

  var yann_dir = 'doc'; // directory containing theanodoc
  var root_url = url.substring(0, url.search('/' + yann_dir)) + '/';

  function build_vswitch() {
    var vlabel = "More";

    var vswitch = ['<div class="rst-versions" data-toggle="rst-versions" role="note" aria-label="versions" align=left>'];
    vswitch.push('<span class="rst-current-version" data-toggle="rst-current-version">');
    vswitch.push('<span class="fa fa-book"></span>');
    vswitch.push(vlabel, '  ');
    vswitch.push('<span class="fa fa-caret-down"></span>');   
    vswitch.push('</span>');
    
    vswitch.push('<div class="rst-other-versions">');   

    vswitch.push('<dt>Downloads</dt>');
    var pdf_url = root_url + "/yann.pdf"
    vswitch.push('<dd><a href=\"', pdf_url, '\">', 'PDF', '</a></dd>');
    vswitch.push('</dl>');  
    
    vswitch.push('<dl>');   
    vswitch.push('<dt>On GitHub</dt>');
    var git_master = "https://github.com/ragavvenkatesan/yann"
    vswitch.push('<dd><a href=\"', git_master + '\">', 'Fork me', '</a></dd>');
    vswitch.push('</dl>');  

    vswitch.push('<dl>');   
    vswitch.push('<dt>Creator</dt>');
    var git_master = "http://www.ragav.net"
    vswitch.push('<dd><a href=\"', "http://www.ragav.net" + '\">', 'Ragav Venkatesan', '</a></dd>');
    vswitch.push('</dl>');  

    vswitch.push('</div>');    
    return vswitch.join('');
  }

// Create HTML for version switcher and assign to placeholder in layout.html.
  $(document).ready(function() {
    // Build default switcher
    $('.version_switcher_placeholder').html(build_vswitch());

    // Check server for other doc versions and update switcher.
    if (url.startsWith('http')) {
      $.getJSON(root_url + 'yann_versions/versions.json', function(data){
        $.each(data, function(version, dir) {
            versions_dir[version] = dir;
        });
        $('.version_switcher_placeholder').html(build_vswitch()); 
      });
    }    
  });
})();
