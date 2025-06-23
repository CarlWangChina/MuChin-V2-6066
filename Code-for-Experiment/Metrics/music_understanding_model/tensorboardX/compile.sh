DESIRED_PROTO_VERSION="3.6.1"

if [ -f "protoc/bin/protoc" ]; then
  PROTOC_BIN="protoc/bin/protoc"
else
  PROTOC_BIN=`which protoc`
fi

echo "using" $PROTOC_BIN

CURRENT_PROTOC_VER=`${PROTOC_BIN} --version`
if [ -z ${PROTOC_BIN} ] || [[ "$CURRENT_PROTOC_VER" != "libprotoc "$DESIRED_PROTO_VERSION ]]; then
  if [ "$(uname)" == "Darwin" ]; then
    PROTOC_ZIP="protoc-"$DESIRED_PROTO_VERSION"-osx-x86_64.zip"
  else
    PROTOC_ZIP="protoc-"$DESIRED_PROTO_VERSION"-linux-x86_64.zip"
  fi
  WGET_BIN=`which wget`
  if [[ ! -z ${WGET_BIN} ]]; then
    ${WGET_BIN} https://github.com/protocolbuffers/protobuf/releases/download/v"$DESIRED_PROTO_VERSION"/${PROTOC_ZIP}
    rm -rf protoc
    python -c "import zipfile; zipfile.ZipFile('"${PROTOC_ZIP}"','r').extractall('protoc')"
    PROTOC_BIN=protoc/bin/protoc
    chmod +x ${PROTOC_BIN}
  fi
fi

if [[ ! -z ${PROTOC_BIN} ]]; then

else
fi
