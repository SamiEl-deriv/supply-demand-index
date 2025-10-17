

#!/etc/rmg/bin/perl
package t::Validation::Transaction::Payment::Deposit;
package BOM::Test::Helper::FinancialAssessment;

use strict;
use warnings;

use Test::More;
use Test::Exception;
use Test::MockObject::Extends;

use BOM::User;
use BOM::User::Client;
use BOM::User::Password;
use BOM::Platform::Client::IDAuthentication;
use Getopt::Long qw(GetOptions);
use Locale::Country::Extra;
use BOM::Platform::Context::Request;

use BOM::RPC::v3::Accounts;
use BOM::Test::Helper::FinancialAssessment;

use BOM::Test::RPC::Client;
use Test::More;
use Test::Mojo;
use BOM::Database::Model::OAuth;
use BOM::Test::RPC::QueueClient;

use Crypt::CBC;
use Crypt::NamedKeys;
Crypt::NamedKeys::keyfile '/etc/rmg/aes_keys.yml';

my $error_message = "Parameters are missing";
my $email = $ARGV[0] or die $error_message;
my $password = $ARGV[1] or die $error_message;
my $broker_code = $ARGV[2] or die $error_message;
my $residence = $ARGV[3] or die $error_message;
my $currency = $ARGV[4] or die $error_message;

my $allow_copier = 1;
my $countries = Locale::Country::Extra->new();

my $method = 'document_upload';
my $file_id;
my $c = BOM::Test::RPC::QueueClient->new();
my $p2p_advertiser_nickname;
my $is_advertiser='dummy_purpose';

GetOptions(
    'pa'              => \my $has_payment_agent,
    'no_deposit'      => \my $no_deposit,
    'no_currency'     => \my $no_currency,
    'copier'          => \my $is_copier,
    'advertiser:s'    => \$is_advertiser,
    'no_ad'           => \my $no_ad,
    'authenticated'   => \my $authenticated,
    'rf'              => \my $is_rf,
    'mt5'             => \my $has_mt5,
    'low_balance'     => \my $low_balance,

);

if ($is_copier) {
    $allow_copier = 0;
};

my @randstr = ("A".."Z", "a".."z");
my $randstr;
$randstr .= $randstr[rand @randstr] for 1..3;

my @randnum = ("0".."9");
my $randnum;
$randnum .= $randnum[rand @randnum] for 1..5;

my $phrase;
if ($password =~ /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=\S{8,})/) {
   $phrase = $password;
} else {
   die "Please change to a more secure phrase in order to proceed";
}

my $name = $email;
$name =~ s/\@.*//;
$name =~ s/[^a-zA-Z,]//g;

my $phone = "+624175".$randnum;
my $last_name = $name.$randstr;

my $hash_pwd = BOM::User::Password::hashpw($phrase);
my $secret_answer = Crypt::NamedKeys->new(keyname => 'client_secret_answer')->encrypt_payload(data => "blah");

my $float = BOM::Config::P2P::advert_config()->{"$residence"}{float_ads};
my $float_rate = 0;

if (defined $float && $float eq "enabled") {                                   
    $float_rate    = 1;
}

my $user = BOM::User->create(
    email    => $email,
    password => $hash_pwd,
    email_verified => 1,
    email_consent => 1
);

my $client_details = {
    broker_code     => $broker_code,
    residence       => $residence,
    client_password => 'x',
    last_name       => $last_name.'ta',
    first_name      => 'QA script',
    email           => $email,
    salutation      => 'Ms',
    address_line_1  => 'ADDR 1',
    address_city    => 'Cyber',
    phone           => $phone,
    citizen         => $residence,
    secret_question => "Mother's maiden name",
    secret_answer   => $secret_answer,
    place_of_birth  => $residence,
    account_opening_reason => 'Speculative',
    date_of_birth => '1990-01-01',
    source          => 1098,
    non_pep_declaration_time => time
};

my $deposit_amount = 1000000;
my @fiat = ( "USD", "EUR", "AUD", "GBP" );

if ($low_balance) {
# Deposit 1000 for low_balance account
     $deposit_amount = 1000000;
}
if (not grep $_ eq $currency, @fiat) {
# Deposit 10 for crypto account
      $deposit_amount = 1000000;
}

my %deposit = (
    currency     => 'USD',
    amount       => 1000000,
    remark       => 'here is money (account created by script)',
    payment_type => 'free_gift'
);


# create virtual account
sub create_virtual{
    my $vrtc_client = $user->create_client(
            broker_code        => 'VRTC',
            first_name         => '',
            email              => $email,
            last_name          => '',
            client_password    => 'x',
            residence          => $residence,
            address_line_1     => '',
            address_line_2     => '',
            address_city       => '',
            address_state      => '',
            address_postcode   => '',
            phone              => '',
            secret_question    => '',
            secret_answer      => ''
    );

    $vrtc_client->email($email);
    $vrtc_client->set_default_account('USD');
    $vrtc_client->payment_free_gift(%deposit, currency => 'USD', amount => 1000000);
    $vrtc_client->save();

    my $broker_code = "VRTC";
    print_login_id($vrtc_client);
    create_api_token($vrtc_client);

}

# create CR account
sub create_cr{
    my $cr_client;
    if ($email =~ /cr_mis_det/) {
        # for rf test account only
        $cr_client = $user->create_client(
            %$client_details,
            address_postcode => '47120',
            place_of_birth => '',
        );
    }
    else{
        $cr_client = $user->create_client(
            %$client_details,
            address_postcode => '47120',
            # allow copier
            allow_copiers => $allow_copier,
        );
    }

    $cr_client->email($email);
    if (!($no_currency)) {
        set_currency($cr_client);
    };
    if (!($no_currency) && !($no_deposit)) {
        deposit($cr_client);
    };
    approve_tnc($cr_client);

    $cr_client->save();

    create_api_token($cr_client);

# ---
    if ($is_rf) {
        my $btc_client = $user->create_client(
            %$client_details
        );

        $btc_client->email($email);
        $btc_client->set_default_account('BTC');
        $btc_client->payment_free_gift(%deposit, currency => $currency, amount => 5);
        $btc_client->user->set_tnc_approval;
        $btc_client->save();

        1;

        my $ltc_client = $user->create_client(
            %$client_details,
        );

        $ltc_client->email($email);
        $ltc_client->set_default_account('LTC');
        $ltc_client->payment_free_gift(%deposit, currency => $currency, amount => 5);
        $ltc_client->user->set_tnc_approval;
        $ltc_client->save();

        1;

        if ($email =~ /check_limit/) {
            my $eth_client = $user->create_client(
            %$client_details
            );

            $eth_client->email($email);
            $eth_client->set_default_account('ETH');
            $eth_client->payment_free_gift(%deposit, currency => $currency, amount => 5);
            $eth_client->user->set_tnc_approval;
            $eth_client->save();

            1;

            my $usdc_client = $user->create_client(
            %$client_details,
            );

            $usdc_client->email($email);
            $usdc_client->set_default_account('USDC');
            $usdc_client->payment_free_gift(%deposit, currency => $currency, amount => 5);
            $usdc_client->user->set_tnc_approval;
            $usdc_client->save();

            1;
        }
    }
# ---

    if ($has_payment_agent) {
        # upload document and authenticate account
        my $file_id;
        my $checksum = 'FileChecksum';

        $file_id = start_successful_upload($cr_client);
        finish_successful_upload($cr_client,$file_id, $checksum);

        payment_agent($cr_client);

        if ($email =~ /pa_restriction/) {
            my $eth_client = $user->create_client(
            %$client_details
            );

            $eth_client->email($email);
            $eth_client->set_default_account('ETH');
            $eth_client->payment_free_gift(%deposit, currency => $currency, amount => 5);
            $eth_client->user->set_tnc_approval;
            $eth_client->save();

            payment_agent($eth_client);

            1;
        }
    };

    if ($is_advertiser ne 'dummy_purpose') {
        if ($is_advertiser ne ''){
            $p2p_advertiser_nickname = $is_advertiser;
        }else{
            $p2p_advertiser_nickname = 'client '.$cr_client->loginid;
        };

        $cr_client->status->set('age_verification', 'system', 'verified using QA script');
        $cr_client->p2p_advertiser_create(name => $p2p_advertiser_nickname);
        $cr_client->p2p_advertiser_update(
            is_listed   => 1,
            is_approved => 1);
        
        if(not $no_ad){
            $cr_client->p2p_advert_create(
            account_currency => 'USD',
            local_currency   => $cr_client->local_currency,
            amount           => 100,
            rate             => 14500,
            type             => 'buy',
            expiry           => 2 * 60 * 60,
            min_order_amount => 0.1,
            max_order_amount => 50,
            payment_method   => 'bank_transfer',
            description      => 'Created by script. Please call me 02203400',
            country          => $cr_client->residence,
            $float_rate
            ? (
                rate      => -0.1,
                rate_type => 'float'
                )
            : (
                rate      => 13500,
                rate_type => 'fixed'
            ),
            );
    
            $cr_client->p2p_advert_create(
            account_currency => 'USD',
            local_currency   => $cr_client->local_currency,
            amount           => 50,
            rate             => 13500,
            type             => 'sell',
            expiry           => 2 * 60 * 60,
            min_order_amount => 0.1,
            max_order_amount => 50,
            payment_method   => 'bank_transfer',
            payment_info     => 'Transfer to account 000-1111',
            contact_info     => 'Created by script. Please call me 02203400',
            description      => 'Created by script. Please call me 02203400',
            country          => $cr_client->residence,
            $float_rate
            ? (
                rate      => -0.1,
                rate_type => 'float'
                )
            : (
                rate      => 13500,
                rate_type => 'fixed'
            ),
            );
        };

    };

    if ($email =~ /zwp2pedit/ || $email =~ /zwp2pbuyerpm/ || $email =~ /zwp2psellerpm/ || $email =~ /zwp2pmobile/ || $email =~ /zwp2pmobile_adv_authenticated/ || $email =~ /seller_p2pmobile/ || $email =~ /buyer_p2pmobile/  ) {
        $cr_client->p2p_advertiser_payment_methods(create => [{
            method    => 'bank_transfer',
            bank_name => 'maybank',
            branch    => '001',
            account   => '1234',
        }]);

        $cr_client->p2p_advertiser_payment_methods(create => [{
            method    => 'cassava_remit',
            account   => '1234',
        }]);

        $cr_client->p2p_advertiser_payment_methods(create => [{
            method    => 'other',
            name      => 'grabby',
            account   => '2123',
        }])
    };

    if ($email =~ /p2pmobile_pn/  ) {
        $cr_client->p2p_advertiser_payment_methods(create => [{
            method    => 'bank_transfer',
            bank_name => 'maybank',
            branch    => '002',
            account   => '1235',
        }]);

        $cr_client->p2p_advertiser_payment_methods(create => [{
            method    => 'other',
            name      => 'grabby',
            account   => '2124',
        }])
    };

    if ($authenticated) {
        # upload document and authenticate account
        my $file_id;
        my $checksum = 'FileChecksum';

        $file_id = start_successful_upload($cr_client);
        finish_successful_upload($cr_client,$file_id, $checksum);

        $cr_client->set_authentication('ID_DOCUMENT', {status => 'pass'});
    };

    print_login_id($cr_client);
    print_residence($cr_client);

}


# create MX account
sub create_mx{
    my $mx_client = $user->create_client(
        %$client_details,
        citizen          => 'gb',
        address_postcode => '47120',
    );

    $mx_client->email($email);
    if (!($no_currency)) {
        set_currency($mx_client);
    };
    if (!($no_currency) && !($no_deposit)) {
        deposit($mx_client);
    };
    approve_tnc($mx_client);
    $mx_client->save();
    $mx_client->status->setnx('unwelcome', 'system', 'FailedExperian - Experian request failed and will be attempted again within 1 hour.');
    $mx_client->status->set('max_turnover_limit_not_set', 'system', 'new GB client or MLT client - have to set turnover limit') ;
    $mx_client->status->setnx('proveid_pending', 'system', 'Experian request failed and will be attempted again within 1 hour.');
    $mx_client->status->setnx('proveid_requested', 'system', 'ProveID request has been made for this account.');

# ---
    if ($is_rf) {
        my $mf_client = $user->create_client(
        %$client_details,
        broker_code     => 'MF',
        tax_residence   => 'gb',
        tax_identification_number => '111-222-333',
        );

        $mf_client->email($email);
        $mf_client->set_default_account($currency);
        $mf_client->payment_free_gift(%deposit, currency => $currency, amount => $deposit_amount);
        $mf_client->user->set_tnc_approval;
        $mf_client->financial_assessment({data => BOM::Test::Helper::FinancialAssessment::mock_maltainvest_fa()});
        $mf_client->save();
    }
# ---

    if ($authenticated) {
        # upload document and authenticate account
        my $file_id;
        my $checksum = 'FileChecksum';

        $file_id = start_successful_upload($mx_client);
        finish_successful_upload($mx_client,$file_id, $checksum);

        $mx_client->set_authentication('ID_DOCUMENT', {status => 'pass'});

    };

    print_login_id($mx_client);
    create_api_token($mx_client);
    print_residence($mx_client);

}

# create MF account
sub create_mf{
    my $mf_client = $user->create_client(
        %$client_details,
        broker_code     => 'MF',
        tax_residence   => 'es',
        tax_identification_number => '111-222-333',
        citizen         => 'es',
    );

    $mf_client->email($email);
    if (!($no_currency)) {
        set_currency($mf_client);
    };
    if (!($no_currency) && !($no_deposit)) {
        deposit($mf_client);
    };
    approve_tnc($mf_client);
    set_financial_assessment($mf_client);
    $mf_client->save();

    if ($authenticated) {
        # upload document and authenticate account
        my $file_id;
        my $checksum = 'FileChecksum';

        $file_id = start_successful_upload($mf_client);
        finish_successful_upload($mf_client,$file_id, $checksum);

        $mf_client->set_authentication('ID_DOCUMENT', {status => 'pass'});
    };

    print_login_id($mf_client);
    create_api_token($mf_client);
    print_residence($mf_client);
}

sub create_mlt{
    my $mlt_client = $user->create_client(
        %$client_details,
        broker_code     => 'MLT',
        citizen         => 'at',
    );

    $mlt_client->email($email);
    if (!($no_currency)) {
        set_currency($mlt_client);
    };
    if (!($no_currency) && !($no_deposit)) {
        deposit($mlt_client);
    };
    approve_tnc($mlt_client);
    $mlt_client->status->set('max_turnover_limit_not_set', 'system', 'new GB client or MLT client - have to set turnover limit') ;
    # when broker=mlt, we will create mf, thus, FA needed in mlt too. Except for belgium which we dont create mf
    if ($residence ne "be" && !($is_rf)){
        set_financial_assessment($mlt_client);
    };
    $mlt_client->save();

    if ($authenticated) {
        # upload document and authenticate account
        my $file_id;
        my $checksum = 'FileChecksum';

        $file_id = start_successful_upload($mlt_client);
        finish_successful_upload($mlt_client,$file_id, $checksum);

        $mlt_client->set_authentication('ID_DOCUMENT', {status => 'pass'});
    };

    print_login_id($mlt_client);
    create_api_token($mlt_client);
    print_residence($mlt_client);

    return $mlt_client;
}

sub start_successful_upload {

    my ($client, $custom_params) = @_;

    my ($real_token)    = BOM::Database::Model::OAuth->new->store_access_token_only(1, $client->loginid);


    my $params = {
        language => 'EN',
        token    => $real_token,
        upload   => 'some_id',
        args     => {
            document_id       => 'ABCD1234',
            document_type     => 'passport',
            document_format   => 'jpg',
            expected_checksum => 'FileChecksum',
            expiration_date   => '2117-08-11',
            file_size         => 1,
            }

        };

    # Call to start upload
    my $result = $c->_tcall($method, $params)->result;

    return $result->{file_id};
}

sub finish_successful_upload {
    my ($client, $file_id, $checksum) = @_;
    my ($real_token)    = BOM::Database::Model::OAuth->new->store_access_token_only(1, $client->loginid);

    my $params = {
    token    => $real_token,
        args     => {
            status  => 'success',
            file_id => $file_id}
    };

    # Call successful upload
    my $result = $c->_tcall($method, $params);

}

sub set_currency{
    my $client = shift;
    $client->set_default_account($currency);
    return $client;
}

sub deposit{
    my $client = shift;
    $client->payment_free_gift(%deposit, currency => $currency, amount => $deposit_amount);
    return $client;
}

sub approve_tnc{
    my $client = shift;
    $client->user->set_tnc_approval;
    my $r = BOM::Platform::Context::Request->new({brand_name => 'binary'});
    BOM::Platform::Context::request($r);
    $client->user->set_tnc_approval;

    return $client;
}

sub maltainvest_fa {
    my %data = (
        "forex_trading_experience"             => "0-1 year",
        "forex_trading_frequency"              => "0-5 transactions in the past 12 months",
        "binary_options_trading_experience"    => "0-1 year",
        "binary_options_trading_frequency"     => "0-5 transactions in the past 12 months",
        "cfd_trading_experience"               => "0-1 year",
        "cfd_trading_frequency"                => "0-5 transactions in the past 12 months",
        "other_instruments_trading_experience" => "0-1 year",
        "other_instruments_trading_frequency"  => "0-5 transactions in the past 12 months",
        "employment_industry"                  => "Health",
        "education_level"                      => "Secondary",
        "income_source"                        => "Self-Employed",
        "net_income"                           => '$25,000 - $50,000',
        "estimated_worth"                      => '$100,000 - $250,000',
        "occupation"                           => 'Managers',
        "employment_status"                    => "Self-Employed",
        "source_of_wealth"                     => "Company Ownership",
        "account_turnover"                     => 'Less than $25,000',
    );

    return encode_json_utf8(\%data);
}

sub set_financial_assessment{
    my $client = shift;
    $client->financial_assessment({data => maltainvest_fa()});
    $client->status->set('financial_risk_approval', 'SYSTEM', 'Client accepted financial risk disclosure');

}

sub create_api_token{
    my $client = shift;
    my $loginid = $client->loginid;
    my $log_broker_code = $loginid;
    $log_broker_code =~ s/\d//g;
    my $res = BOM::RPC::v3::Accounts::api_token({
            client => $client,
            args   => {
                new_token        => 'Created by script',
                new_token_scopes => ['read', 'trade','payments','admin']
            },
        });

    my $token = $res->{tokens}->[0]->{token};
    print "$log_broker_code api token: $token
";
}

sub payment_agent{

  my $client = shift;
  my $loginid = $client->loginid;
  my $currency = $client->currency;
  my $residence = $client->residence;

  $client->set_authentication('ID_DOCUMENT', {status => 'pass'});

       my $payment_agent = {
      payment_agent_name    => 'Payment Agent of '.$loginid.' (Created from Script)',
      urls                  => [{url => 'http://www.MyPAMyAdventure.com/'},{url => 'http://www.MyPAMyAdventure2.com/'}],
      email                 => 'MyPaScript@example.com',
      phone_numbers         => [{phone_number => '+12345678'}],
      information           => 'Test Info',
      summary               => 'Test Summary',
      commission_deposit    => 0,
      commission_withdrawal => 0,
      status                => 'authorized',
      supported_payment_methods       => [{payment_method => 'MasterCard'},{payment_method => 'Visa'}],
      currency_code         => $currency,
      target_country        => $residence,
      min_withdrawal        => '10',
      max_withdrawal        => '2000',
      };
  my $pa_client = $client->set_payment_agent();
     $pa_client->$_($payment_agent->{$_}) for keys %$payment_agent;
     $pa_client->save
}

sub print_login_id{
    my $client = shift;
    my $loginid = $client->loginid;
    my $log_broker_code = $loginid;
    $log_broker_code =~ s/\d//g;

    print "$log_broker_code loginid: $loginid
";
}

sub print_residence{
    my $client = shift;
    my $client_residence = $countries->country_from_code($client->residence);
    my $loginid = $client->loginid;
    my $log_broker_code = $loginid;
    $log_broker_code =~ s/\d//g;

    print "Residence: $client_residence
";
}

sub create_mt5 {

    my $mt5_real = BOM::MT5::User::Async::create_user({
        name           => "QA script - mt5 real",
        mainPassword   => $phrase,
        investPassword => $phrase.$phrase,
        group          => 'real\p01_ts03\synthetic\svg_std_usd\01',
        leverage       => 100
    })->get;

    my $mt5_demo = BOM::MT5::User::Async::create_user({
        name           => "QA script - mt5 demo",
        mainPassword   => $phrase,
        investPassword => $phrase.$phrase,
        group          => 'demo\p01_ts01\synthetic\svg_std_usd',
        leverage       => 100
    })->get;

    if ($mt5_real) {
        $user->add_loginid($mt5_real->{login});
        print "MT5 real loginid: $mt5_real->{login}\n";
    }

    if ($mt5_demo) {
        $user->add_loginid($mt5_demo->{login});
        print "MT5 demo loginid: $mt5_demo->{login}\n";
    }

    $user->update_trading_password($phrase);

}


#------------------------------------------------------

# create VRTC account
if ($broker_code eq "VRTC"){
    create_virtual();
}

# create CR account
if ($broker_code eq "CR"){
    create_virtual();
    create_cr();

    if ($has_mt5) {
        create_mt5();
    }
}

# create MX account
elsif ($broker_code eq "MX"){
    create_virtual();
    create_mx();
}

# create MF account
elsif ($broker_code eq "MF"){
    create_virtual();
    create_mf();
}

# create MLT MF account
elsif ($broker_code eq "MLT"){
    create_virtual();
    create_mlt();
    if ($residence ne "be" && !($is_rf)){
        create_mf();
    }
}

1;
