<?php
if ($argc < 3) 
    {
    echo "usage: php {$argv[0]} lang-tag number ...\n";
    exit;
    }

array_shift($argv);
$lang_tag = array_shift($argv);

$nf1 = new NumberFormatter($lang_tag, NumberFormatter::DECIMAL);
$nf2 = new NumberFormatter($lang_tag, NumberFormatter::SPELLOUT);

foreach ($argv as $num) 
    {
    #echo $nf1->format($num).' is '.$nf2->format($num)."\n"; 
    echo $num.' , '.$nf2->format($num)."\n"; 

    }

?>
