<?php
error_reporting(0);
$output			= '';
define(UPLOAD_ERR_OK, 0);
if (isset($_POST['Submit'])) {
	
	
	$name		=$_POST['textfield'];
	$email		=$_POST['textfield2'];
	$contact	        =$_POST['textfield3'];
	$comment	        =$_POST['textarea'];

	$body		= "<b><u>Contact Me</u></b><br /><br />";
	$body		= $body.'<table border="1" cellpadding="4">';
	$body		= $body."<tr>"."<th>"."Name"."</th>"."<td>".$name."</td>"."</tr>";
	$body		= $body."<tr>"."<th>"."Email"."</th>"."<td>".$email."</td>"."</tr>";
	$body		= $body."<tr>"."<th>"."Contact No"."</th>"."<td>".$contact."</td>"."</tr>";
	$body		= $body."<tr>"."<th>"."Comments"."</th>"."<td>".$comment."</td>"."</tr>";
	$body		= $body."</table>";


require_once('PHPMailer/class.phpmailer.php');
		$mail				= new PHPMailer();
		$mail->IsSMTP();
		$mail->Mailer		= "smtp";
		$mail->SMTPDebug	= 0;
		$mail->SMTPAuth		= true;
		$mail->SMTPSecure	= "ssl";
		$mail->Host			= "smtp.gmail.com";
		$mail->Port			= 465;
	$mail->Username		= "noreply.srikala.gangi.reddy@gmail.com";
	$mail->Password		= "noreplynoreply";		

		$mail->SetFrom('noreply.srikala.gangi.reddy@gmail.com', 'No-Reply');
		

		$mail->AddAddress('srikala.gangi.reddy@gmail.com', 'customercare');
		$mail->AddBcc('senthilkumarkindia@gmail.com', 'srikal');
		$mail->Subject	= 'Contact Form';

	

	if(isset($_POST['cbAsCopy'])) {
		$mail->AddAddress($_POST['email']);		
	}
	$mail->Subject	= 'Contact Form';	
	$mail->MsgHTML($body);

	if(!$mail->Send()) {
	$output	= 	header('Location:http://undertheyellowtree.com/pages/error.html');

	
	} else {

	$output	= 	header('Location:http://undertheyellowtree.com/pages/thankyou.html');
	}
}
echo $output;
?>
