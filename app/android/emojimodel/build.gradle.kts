plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.framna.emojimodel"
    compileSdk = 37

    defaultConfig {
        minSdk = 28
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    // model.bin is already dense float32; compressing it only slows the first load.
    androidResources {
        noCompress += "bin"
    }
}

dependencies {
    implementation(libs.kotlin.serialization.json)
    testImplementation(libs.junit)
}

tasks.withType<Test>().configureEach {
    maxHeapSize = "1g"
    testLogging {
        events("passed", "failed")
        showStandardStreams = true
    }
}
