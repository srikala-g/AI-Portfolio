<?php
//$ToEmail = 'info@bigbangoffer.com';
$ToEmail = 'rahulraj@wabsus.com';
$EmailSubject = 'BigBangOffer Contact Us';
$mailheader = "From: ".$_POST["textfield2"]."\r\n";
$mailheader .= "Reply-To: ".$_POST["textfield2"]."\r\n";
$mailheader .= "Content-Type: text/plain; charset=us-ascii";
$MESSAGE_BODY .= "Name	: ".$_POST["textfield"]."\r\n\n";
$MESSAGE_BODY .= "Email	: ".$_POST["textfield2"]."\r\n\n";
$MESSAGE_BODY .= "Contact No	: ".$_POST["textfield3"]."\r\n\n";
$MESSAGE_BODY .= "Comments	: ".$_POST["textarea"]."\r\n\n";
mail($ToEmail, $EmailSubject, $MESSAGE_BODY, $mailheader) or die ("Failure");
header('Location:http://undertheyellowtree.com/pages/thankyou.html');
?>