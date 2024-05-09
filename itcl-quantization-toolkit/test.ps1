$test_directories = @( ".\core", ".\inference_engine\itcl_inference_engine", ".\quantizer\itcl_quantizer")

$test_directories = @( ".\inference_engine\itcl_inference_engine", ".\quantizer\itcl_quantizer")

$root_dir = $PSScriptRoot

foreach ($dir in $test_directories){
    Set-Location $dir

    if(![System.IO.Directory]::Exists(".venv") -eq $false)
    {
        poetry install
    }

    poetry run pytest
    Set-Location $root_dir
}
